"""Compare Klay's PyTorch-style sampler with SpringMV's JAX sampler.

The benchmark supplies identical resident log weights, Gumbels, and parent
activity to both implementations. RNG generation, transfers, compilation,
and CSR preprocessing are intentionally outside the timed region.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

import springmv


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return sizes


def _summarize(samples: Sequence[float]) -> tuple[float, float, float]:
    return statistics.median(samples), min(samples), max(samples)


def _time_jax(operation: Callable[[], object]) -> float:
    start = time.perf_counter()
    jax.block_until_ready(operation())
    return (time.perf_counter() - start) * 1.0e3


def _time_torch(operation: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1.0e3


def _measure(
    operation: Callable[[], object],
    timer: Callable[[Callable[[], object]], float],
    warmup: int,
    iterations: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        timer(operation)
    return _summarize([timer(operation) for _ in range(iterations)])


def _format_timing(samples: tuple[float, float, float]) -> str:
    median, minimum, maximum = samples
    return f"{median:9.4f} [{minimum:9.4f}, {maximum:9.4f}] ms"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--nnz-per-row", type=int, default=8)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=[1, 128, 1024],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--batched-weights",
        action="store_true",
        help="use distinct log weights for every sample",
    )
    parser.add_argument(
        "--no-torch-compile",
        action="store_true",
        help="skip the torch.compile comparison",
    )
    args = parser.parse_args()

    for name in ("rows", "cols", "nnz_per_row", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("the benchmark requires CUDA-enabled PyTorch")
    try:
        jax_devices = jax.devices("gpu")
    except RuntimeError as error:
        parser.error(f"the benchmark requires CUDA-enabled JAX: {error}")
    if not jax_devices:
        parser.error("the benchmark requires CUDA-enabled JAX")

    rng = np.random.default_rng(args.seed)
    nnz = args.rows * args.nnz_per_row
    if nnz > np.iinfo(np.int32).max:
        parser.error("the requested matrix has too many nonzeros for int32")
    row_offsets = np.arange(
        0, nnz + 1, args.nnz_per_row, dtype=np.int32
    )
    column_indices = rng.integers(
        0, args.cols, size=nnz, dtype=np.int32
    )
    edge_rows = np.repeat(
        np.arange(args.rows, dtype=np.int64), args.nnz_per_row
    )
    structure = springmv.CsrStructure.from_arrays(
        row_offsets, column_indices, num_cols=args.cols
    )

    torch_device = torch.device("cuda")
    torch_offsets = torch.as_tensor(
        row_offsets.astype(np.int64), device=torch_device
    )
    torch_columns = torch.as_tensor(
        column_indices.astype(np.int64), device=torch_device
    )
    torch_edge_rows = torch.as_tensor(edge_rows, device=torch_device)
    shared_weights = rng.normal(size=nnz).astype(np.float32)

    print(
        f"rows={args.rows} cols={args.cols} nnz={nnz} "
        f"nnz/row={args.nnz_per_row} "
        f"batched-weights={args.batched_weights}"
    )
    print(
        "times are median [minimum, maximum]; ratios are "
        "PyTorch/SpringMV"
    )
    print("Gumbel generation, transfers, and compilation are not timed")

    for batch_size in args.batch_sizes:
        uniform = rng.uniform(
            1.0e-6,
            1.0 - 1.0e-6,
            size=(batch_size, nnz),
        ).astype(np.float32)
        gumbels = -np.log(-np.log(uniform)).astype(np.float32)
        active_parent = rng.random((batch_size, args.rows)) < 0.5
        if args.batched_weights:
            scales = np.linspace(0.9, 1.1, batch_size, dtype=np.float32)
            log_weights = shared_weights[None, :] * scales[:, None]
            jax_in_axes = (0, 0, 0)
        else:
            log_weights = shared_weights
            jax_in_axes = (None, 0, 0)

        jax_weights = jnp.asarray(log_weights)
        jax_gumbels = jnp.asarray(gumbels)
        jax_active = jnp.asarray(active_parent)
        torch_weights = torch.as_tensor(
            log_weights.T.copy() if args.batched_weights else log_weights,
            device=torch_device,
        )
        torch_gumbels = torch.as_tensor(
            gumbels.T.copy(), device=torch_device
        )
        torch_active = torch.as_tensor(
            active_parent.T.copy(), device=torch_device
        )

        def springmv_one(weights, noise, active):
            return springmv.csr_logsumexp_sample(
                structure, weights, noise, active
            )

        jax_sample = jax.jit(
            jax.vmap(springmv_one, in_axes=jax_in_axes)
        )

        def torch_sample(weights, noise, active):
            scores = noise + (
                weights if weights.dim() == 2 else weights.unsqueeze(1)
            )
            row_max = torch.segment_reduce(
                scores,
                "max",
                offsets=torch_offsets,
                axis=0,
                unsafe=True,
            )
            selected = scores == row_max[torch_edge_rows]
            selected &= active[torch_edge_rows]
            counts = torch.zeros(
                (args.cols, batch_size),
                dtype=torch.int32,
                device=torch_device,
            )
            child_indices = torch_columns[:, None].expand_as(selected)
            active_child = torch.scatter_reduce(
                counts,
                0,
                index=child_indices,
                src=selected.to(torch.int32),
                reduce="sum",
                include_self=False,
            )
            return (active_child > 0).mT

        jax_bound = lambda: jax_sample(
            jax_weights, jax_gumbels, jax_active
        )
        torch_bound = lambda: torch_sample(
            torch_weights, torch_gumbels, torch_active
        )

        jax_result = jax.block_until_ready(jax_bound())
        torch_result = torch_bound()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(
            np.asarray(jax_result), torch_result.cpu().numpy()
        )

        springmv_times = _measure(
            jax_bound, _time_jax, args.warmup, args.iterations
        )
        torch_eager_times = _measure(
            torch_bound, _time_torch, args.warmup, args.iterations
        )
        springmv_median = springmv_times[0]
        print(
            f"batch={batch_size:5d}  "
            f"SpringMV/JAX {_format_timing(springmv_times)}  "
            f"PyTorch eager {_format_timing(torch_eager_times)}  "
            f"ratio {torch_eager_times[0] / springmv_median:7.2f}x"
        )

        if not args.no_torch_compile:
            try:
                compiled_sample = torch.compile(
                    torch_sample, mode="reduce-overhead"
                )
                compiled_bound = lambda: compiled_sample(
                    torch_weights, torch_gumbels, torch_active
                )
                compiled_result = compiled_bound()
                torch.cuda.synchronize()
                np.testing.assert_array_equal(
                    np.asarray(jax_result), compiled_result.cpu().numpy()
                )
                compiled_times = _measure(
                    compiled_bound,
                    _time_torch,
                    args.warmup,
                    args.iterations,
                )
                print(
                    f"{'':13s}  "
                    f"{'':46s}  "
                    f"PyTorch compiled {_format_timing(compiled_times)}  "
                    f"ratio {compiled_times[0] / springmv_median:7.2f}x"
                )
            except Exception as error:  # noqa: BLE001
                print(
                    f"{'':13s}  torch.compile unavailable: "
                    f"{type(error).__name__}: {error}"
                )


if __name__ == "__main__":
    main()
