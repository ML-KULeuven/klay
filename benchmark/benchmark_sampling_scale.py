"""Measure time and GPU memory for a large number of sampler draws.

SpringMV/JAX and Klay's PyTorch sampling algorithm run in separate worker
processes. Gumbel generation and active-parent generation are included;
topology construction, transfers, warm-up, and compilation are excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np


_RESULT_PREFIX = "SAMPLING_SCALE_RESULT="


def _topology(
    rows: int,
    cols: int,
    nnz_per_row: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nnz = rows * nnz_per_row
    row_offsets = np.arange(0, nnz + 1, nnz_per_row, dtype=np.int32)
    column_indices = rng.integers(0, cols, size=nnz, dtype=np.int32)
    edge_rows = np.repeat(
        np.arange(rows, dtype=np.int32), nnz_per_row
    )
    permutation = np.argsort(column_indices, kind="stable").astype(
        np.int32
    )
    transpose_columns = edge_rows[permutation]
    transpose_counts = np.bincount(
        column_indices, minlength=cols
    ).astype(np.int32)
    transpose_offsets = np.empty(cols + 1, dtype=np.int32)
    transpose_offsets[0] = 0
    np.cumsum(transpose_counts, out=transpose_offsets[1:])
    return {
        "row_offsets": row_offsets,
        "column_indices": column_indices,
        "edge_rows": edge_rows,
        "transpose_row_offsets": transpose_offsets,
        "transpose_column_indices": transpose_columns,
        "transpose_permutation": permutation,
        "log_weights": rng.normal(size=nnz).astype(np.float32),
    }


def _batch_plan(total_samples: int, chunk_size: int) -> list[int]:
    full_chunks, remainder = divmod(total_samples, chunk_size)
    plan = [chunk_size] * full_chunks
    if remainder:
        plan.append(remainder)
    return plan


def _jax_worker(args: argparse.Namespace) -> dict[str, int | float | str]:
    import jax
    import jax.numpy as jnp

    import springmv

    arrays = _topology(
        args.rows,
        args.cols,
        args.nnz_per_row,
        args.seed,
    )
    nnz = args.rows * args.nnz_per_row
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
        num_nonzeros=nnz,
        max_nnz_per_row=args.nnz_per_row,
        transpose_max_nnz_per_row=int(
            np.max(
                np.diff(arrays["transpose_row_offsets"]), initial=0
            )
        ),
    )
    log_weights = jnp.asarray(arrays["log_weights"])

    def make_operation(batch_size: int):
        def sample_one(gumbels, active_parent):
            return springmv.csr_logsumexp_sample(
                structure,
                log_weights,
                gumbels,
                active_parent,
            )

        batched_sample = jax.vmap(sample_one)

        @jax.jit
        def operation(key):
            next_key, uniform_key, active_key = jax.random.split(key, 3)
            uniform = jax.random.uniform(
                uniform_key,
                (batch_size, nnz),
                dtype=jnp.float32,
                minval=jnp.float32(1.0e-6),
                maxval=jnp.float32(1.0 - 1.0e-6),
            )
            gumbels = -jnp.log(-jnp.log(uniform))
            active_parent = jax.random.bernoulli(
                active_key,
                p=0.5,
                shape=(batch_size, args.rows),
            )
            samples = batched_sample(gumbels, active_parent)
            return next_key, samples

        return operation

    plan = _batch_plan(args.total_samples, args.chunk_size)
    operations = {
        batch_size: make_operation(batch_size)
        for batch_size in set(plan)
    }
    key = jax.random.key(args.seed)
    for batch_size, operation in operations.items():
        key, samples = operation(key)
        jax.block_until_ready((key, samples))
    jax.block_until_ready((structure, log_weights, key))
    device = jax.devices("gpu")[0]
    before = device.memory_stats()
    start = time.perf_counter()
    for batch_size in plan:
        key, samples = operations[batch_size](key)
    jax.block_until_ready((key, samples))
    elapsed = time.perf_counter() - start
    after = device.memory_stats()
    checksum = int(jnp.sum(samples, dtype=jnp.int32))
    resident = int(before["bytes_in_use"])
    peak_live = int(after["peak_bytes_in_use"])
    return {
        "backend": args.worker,
        "elapsed_seconds": elapsed,
        "samples_per_second": args.total_samples / elapsed,
        "checksum": checksum,
        "resident": resident,
        "peak_live": peak_live,
        "incremental_peak": max(0, peak_live - resident),
        "peak_pool": int(after.get("peak_pool_bytes", 0)),
    }


def _torch_worker(
    args: argparse.Namespace,
) -> dict[str, int | float | str]:
    import gc

    import torch

    arrays = _topology(
        args.rows,
        args.cols,
        args.nnz_per_row,
        args.seed,
    )
    device = torch.device("cuda")
    nnz = args.rows * args.nnz_per_row
    offsets = torch.as_tensor(
        arrays["row_offsets"].astype(np.int64), device=device
    )
    columns = torch.as_tensor(
        arrays["column_indices"].astype(np.int64), device=device
    )
    edge_rows = torch.as_tensor(
        arrays["edge_rows"].astype(np.int64), device=device
    )
    log_weights = torch.as_tensor(
        arrays["log_weights"], device=device
    )

    def operation(batch_size: int):
        uniform = torch.rand(
            (nnz, batch_size), dtype=torch.float32, device=device
        )
        uniform.clamp_(min=1.0e-6, max=1.0 - 1.0e-6)
        gumbels = -torch.log(-torch.log(uniform))
        active_parent = torch.rand(
            (args.rows, batch_size),
            dtype=torch.float32,
            device=device,
        ) < 0.5
        scores = gumbels + log_weights[:, None]
        row_max = torch.segment_reduce(
            scores,
            "max",
            offsets=offsets,
            axis=0,
            unsafe=True,
        )
        selected = scores == row_max[edge_rows]
        selected &= active_parent[edge_rows]
        counts = torch.zeros(
            (args.cols, batch_size),
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
        return active_child > 0

    plan = _batch_plan(args.total_samples, args.chunk_size)
    torch.manual_seed(args.seed)
    for batch_size in set(plan):
        operation(batch_size)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = int(torch.cuda.memory_allocated())
    start = time.perf_counter()
    for batch_size in plan:
        samples = operation(batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_live = int(torch.cuda.max_memory_allocated())
    checksum = int(torch.sum(samples, dtype=torch.int64))
    return {
        "backend": args.worker,
        "elapsed_seconds": elapsed,
        "samples_per_second": args.total_samples / elapsed,
        "checksum": checksum,
        "resident": resident,
        "peak_live": peak_live,
        "incremental_peak": max(0, peak_live - resident),
        "peak_pool": int(torch.cuda.max_memory_reserved()),
    }


def _worker(args: argparse.Namespace) -> None:
    if args.worker == "springmv-jax":
        result = _jax_worker(args)
    else:
        result = _torch_worker(args)
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))


def _run_worker(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "benchmark.benchmark_sampling_scale",
        "--worker",
        backend,
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--nnz-per-row",
        str(args.nnz_per_row),
        "--total-samples",
        str(args.total_samples),
        "--chunk-size",
        str(args.chunk_size),
        "--seed",
        str(args.seed),
    ]
    environment = os.environ.copy()
    if backend == "springmv-jax":
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
        f"{backend} worker returned no result:\n"
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
    nnz = args.rows * args.nnz_per_row
    chunks = len(_batch_plan(args.total_samples, args.chunk_size))
    print(
        f"rows={args.rows} cols={args.cols} nnz={nnz} "
        f"nnz/row={args.nnz_per_row} samples={args.total_samples} "
        f"chunk={args.chunk_size} chunks={chunks}"
    )
    print(
        "Gumbel and active-parent generation are included; construction, "
        "transfers, warm-up, and compilation are excluded"
    )
    print(
        f"{'backend':18s} {'time':>10s} {'samples/s':>14s} "
        f"{'resident':>12s} {'peak live':>12s} "
        f"{'extra peak':>12s} {'peak pool':>12s}"
    )
    results = []
    for backend in ("springmv-jax", "klay-pytorch"):
        result = _run_worker(args, backend)
        results.append(result)
        print(
            f"{backend:18s} "
            f"{result['elapsed_seconds']:9.3f}s "
            f"{result['samples_per_second']:14,.0f} "
            f"{_format_bytes(result['resident']):>12s} "
            f"{_format_bytes(result['peak_live']):>12s} "
            f"{_format_bytes(result['incremental_peak']):>12s} "
            f"{_format_bytes(result['peak_pool']):>12s}"
        )
    ratio = (
        results[0]["elapsed_seconds"] / results[1]["elapsed_seconds"]
    )
    print(
        "time ratio SpringMV/JAX / Klay/PyTorch: "
        f"{ratio:.3f}x "
        f"({1.0 / ratio:.2f}x SpringMV throughput)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--nnz-per-row", type=int, default=8)
    parser.add_argument("--total-samples", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--worker",
        choices=("springmv-jax", "klay-pytorch"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    for name in (
        "rows",
        "cols",
        "nnz_per_row",
        "total_samples",
        "chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    nnz = args.rows * args.nnz_per_row
    if nnz > np.iinfo(np.int32).max:
        parser.error("the requested matrix has too many nonzeros for int32")
    if args.worker:
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
