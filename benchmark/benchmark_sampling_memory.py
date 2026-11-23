"""Compare sampler GPU memory use in isolated JAX and PyTorch processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np


_RESULT_PREFIX = "SAMPLING_MEMORY_RESULT="


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


def _arrays(
    rows: int,
    cols: int,
    nnz_per_row: int,
    batch_size: int,
    seed: int,
    batched_weights: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nnz = rows * nnz_per_row
    row_offsets = np.arange(0, nnz + 1, nnz_per_row, dtype=np.int32)
    column_indices = rng.integers(0, cols, size=nnz, dtype=np.int32)
    edge_rows = np.repeat(
        np.arange(rows, dtype=np.int64), nnz_per_row
    )
    transpose_permutation = np.argsort(
        column_indices, kind="stable"
    ).astype(np.int32)
    transpose_column_indices = edge_rows[transpose_permutation].astype(
        np.int32
    )
    transpose_counts = np.bincount(
        column_indices, minlength=cols
    ).astype(np.int32)
    transpose_row_offsets = np.empty(cols + 1, dtype=np.int32)
    transpose_row_offsets[0] = 0
    np.cumsum(transpose_counts, out=transpose_row_offsets[1:])
    shared_weights = rng.normal(size=nnz).astype(np.float32)
    if batched_weights:
        scales = np.linspace(0.9, 1.1, batch_size, dtype=np.float32)
        log_weights = shared_weights[None, :] * scales[:, None]
    else:
        log_weights = shared_weights
    uniform = rng.uniform(
        1.0e-6, 1.0 - 1.0e-6, size=(batch_size, nnz)
    ).astype(np.float32)
    gumbels = -np.log(-np.log(uniform)).astype(np.float32)
    active_parent = rng.random((batch_size, rows)) < 0.5
    return {
        "row_offsets": row_offsets,
        "column_indices": column_indices,
        "edge_rows": edge_rows,
        "transpose_row_offsets": transpose_row_offsets,
        "transpose_column_indices": transpose_column_indices,
        "transpose_permutation": transpose_permutation,
        "log_weights": log_weights,
        "gumbels": gumbels,
        "active_parent": active_parent,
    }


def _jax_worker(args: argparse.Namespace) -> dict[str, int | str]:
    import jax
    import jax.numpy as jnp

    arrays = _arrays(
        args.rows,
        args.cols,
        args.nnz_per_row,
        args.batch_size,
        args.seed,
        args.batched_weights,
    )
    device = jax.devices("gpu")[0]
    weights = jnp.asarray(arrays["log_weights"])
    gumbels = jnp.asarray(arrays["gumbels"])
    active_parent = jnp.asarray(arrays["active_parent"])
    in_axes = (0, 0, 0) if args.batched_weights else (None, 0, 0)

    if args.worker.startswith("springmv"):
        import springmv

        structure = springmv.CsrStructure(
            row_offsets=jnp.asarray(arrays["row_offsets"]),
            column_indices=jnp.asarray(arrays["column_indices"]),
            transpose_row_offsets=jnp.asarray(
                arrays["transpose_row_offsets"]
            ),
            transpose_column_indices=jnp.asarray(
                arrays["transpose_column_indices"]
            ),
            transpose_permutation=jnp.asarray(
                arrays["transpose_permutation"]
            ),
            num_rows=args.rows,
            num_cols=args.cols,
            num_nonzeros=args.rows * args.nnz_per_row,
            max_nnz_per_row=args.nnz_per_row,
            transpose_max_nnz_per_row=int(
                np.max(
                    np.diff(arrays["transpose_row_offsets"]), initial=0
                )
            ),
        )

        def sample_one(current_weights, current_gumbels, current_active):
            return springmv.csr_logsumexp_sample(
                structure,
                current_weights,
                current_gumbels,
                current_active,
            )

        ready_inputs = (
            structure,
            weights,
            gumbels,
            active_parent,
        )
    else:
        edge_rows = jnp.asarray(
            arrays["edge_rows"].astype(np.int32, copy=False)
        )
        column_indices = jnp.asarray(arrays["column_indices"])

        def sample_one(current_weights, current_gumbels, current_active):
            scores = current_weights + current_gumbels
            row_max = jax.ops.segment_max(
                scores,
                edge_rows,
                num_segments=args.rows,
                indices_are_sorted=True,
            )
            selected = scores == row_max[edge_rows]
            selected &= current_active[edge_rows]
            child_counts = jax.ops.segment_sum(
                selected.astype(jnp.int32),
                column_indices,
                num_segments=args.cols,
            )
            return child_counts > 0

        ready_inputs = (
            edge_rows,
            column_indices,
            weights,
            gumbels,
            active_parent,
        )

    operation = jax.jit(jax.vmap(sample_one, in_axes=in_axes))
    executable = operation.lower(weights, gumbels, active_parent).compile()
    jax.block_until_ready(ready_inputs)
    before = device.memory_stats()
    result = executable(weights, gumbels, active_parent)
    jax.block_until_ready(result)
    after = device.memory_stats()
    resident = int(before["bytes_in_use"])
    peak_live = int(after["peak_bytes_in_use"])
    return {
        "backend": args.worker,
        "resident": resident,
        "after_live": int(after["bytes_in_use"]),
        "peak_live": peak_live,
        "incremental_peak": max(0, peak_live - resident),
        "pool": int(after.get("pool_bytes", 0)),
        "peak_pool": int(after.get("peak_pool_bytes", 0)),
    }


def _torch_worker(args: argparse.Namespace) -> dict[str, int | str]:
    import gc

    import torch

    arrays = _arrays(
        args.rows,
        args.cols,
        args.nnz_per_row,
        args.batch_size,
        args.seed,
        args.batched_weights,
    )
    device = torch.device("cuda")
    offsets = torch.as_tensor(
        arrays["row_offsets"].astype(np.int64), device=device
    )
    columns = torch.as_tensor(
        arrays["column_indices"].astype(np.int64), device=device
    )
    edge_rows = torch.as_tensor(arrays["edge_rows"], device=device)
    weights = torch.as_tensor(
        (
            arrays["log_weights"].T.copy()
            if args.batched_weights
            else arrays["log_weights"]
        ),
        device=device,
    )
    gumbels = torch.as_tensor(arrays["gumbels"].T.copy(), device=device)
    active_parent = torch.as_tensor(
        arrays["active_parent"].T.copy(), device=device
    )

    def sample(current_weights, current_gumbels, current_active):
        scores = current_gumbels + (
            current_weights
            if current_weights.dim() == 2
            else current_weights.unsqueeze(1)
        )
        row_max = torch.segment_reduce(
            scores,
            "max",
            offsets=offsets,
            axis=0,
            unsafe=True,
        )
        selected = scores == row_max[edge_rows]
        selected &= current_active[edge_rows]
        counts = torch.zeros(
            (args.cols, args.batch_size),
            dtype=torch.int32,
            device=device,
        )
        child_indices = columns[:, None].expand_as(selected)
        active_child = torch.scatter_reduce(
            counts,
            0,
            index=child_indices,
            src=selected.to(torch.int32),
            reduce="sum",
            include_self=False,
        )
        return (active_child > 0).mT

    operation = (
        torch.compile(sample, mode="reduce-overhead")
        if args.worker == "torch-compiled"
        else sample
    )
    warm = operation(weights, gumbels, active_parent)
    torch.cuda.synchronize()
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = int(torch.cuda.memory_allocated())
    result = operation(weights, gumbels, active_parent)
    torch.cuda.synchronize()
    peak_live = int(torch.cuda.max_memory_allocated())
    return {
        "backend": args.worker,
        "resident": resident,
        "after_live": int(torch.cuda.memory_allocated()),
        "peak_live": peak_live,
        "incremental_peak": max(0, peak_live - resident),
        "pool": int(torch.cuda.memory_reserved()),
        "peak_pool": int(torch.cuda.max_memory_reserved()),
    }


def _worker(args: argparse.Namespace) -> None:
    if args.worker.startswith(("springmv", "jax-native")):
        result = _jax_worker(args)
    else:
        result = _torch_worker(args)
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))


def _run_isolated(
    args: argparse.Namespace, backend: str, batch_size: int
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "benchmark.benchmark_sampling_memory",
        "--worker",
        backend,
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--nnz-per-row",
        str(args.nnz_per_row),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(args.seed),
    ]
    if args.batched_weights:
        command.append("--batched-weights")
    environment = os.environ.copy()
    if backend in ("springmv-default", "jax-native-default"):
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    elif backend in (
        "springmv-no-preallocate",
        "jax-native-no-preallocate",
    ):
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{backend} worker failed:\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line.removeprefix(_RESULT_PREFIX))
    raise RuntimeError(
        f"{backend} worker returned no memory result:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _format_bytes(value: int) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or suffix == "TiB":
            return f"{amount:.2f} {suffix}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _parent(args: argparse.Namespace) -> None:
    print(
        f"rows={args.rows} cols={args.cols} "
        f"nnz={args.rows * args.nnz_per_row} "
        f"nnz/row={args.nnz_per_row} "
        f"batched-weights={args.batched_weights}"
    )
    print(
        "resident and peak-live measure tensors owned by the framework; "
        "peak-pool includes allocator reservation"
    )
    print(
        "JAX default preallocation therefore appears in peak-pool, not in "
        "peak-live"
    )
    backends = (
        "springmv-default",
        "springmv-no-preallocate",
        "jax-native-default",
        "jax-native-no-preallocate",
        "torch-eager",
        "torch-compiled",
    )
    for batch_size in args.batch_sizes:
        print(f"\nbatch={batch_size}")
        print(
            f"{'backend':28s} {'resident':>12s} {'peak live':>12s} "
            f"{'extra peak':>12s} {'peak pool':>12s}"
        )
        for backend in backends:
            result = _run_isolated(args, backend, batch_size)
            print(
                f"{backend:28s} "
                f"{_format_bytes(result['resident']):>12s} "
                f"{_format_bytes(result['peak_live']):>12s} "
                f"{_format_bytes(result['incremental_peak']):>12s} "
                f"{_format_bytes(result['peak_pool']):>12s}"
            )


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
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batched-weights", action="store_true")
    parser.add_argument(
        "--worker",
        choices=(
            "springmv-default",
            "springmv-no-preallocate",
            "jax-native-default",
            "jax-native-no-preallocate",
            "torch-eager",
            "torch-compiled",
        ),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    for name in ("rows", "cols", "nnz_per_row"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.worker:
        if args.batch_size is None or args.batch_size <= 0:
            parser.error("worker mode requires a positive --batch-size")
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
