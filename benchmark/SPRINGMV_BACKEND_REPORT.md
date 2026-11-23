# Native JAX vs SpringMV backend benchmark

Date: 2026-11-23

## Summary

SpringMV improves Klay's log-semiring forward performance at every tested
batch size. For combined forward and backward execution, native JAX is faster
at batch size 1, while SpringMV becomes faster as the batch grows.

| Batch | Operation | Native JAX | SpringMV | JAX / SpringMV |
|---:|---|---:|---:|---:|
| 1 | Forward | 0.1850 ms | 0.1474 ms | **1.26x** |
| 1 | Forward + backward | **0.2605 ms** | 0.3002 ms | 0.87x |
| 128 | Forward | 0.2217 ms | 0.1826 ms | **1.21x** |
| 128 | Forward + backward | 0.4091 ms | 0.3770 ms | **1.09x** |
| 1024 | Forward | 0.5887 ms | 0.3086 ms | **1.91x** |
| 1024 | Forward + backward | 1.0341 ms | 0.6378 ms | **1.62x** |

A ratio above 1 means SpringMV is faster. These values are medians over 50
synchronized executions after 10 warm-up executions.

## Workload

All three measurements use the same randomly generated SDD:

- 100 Boolean variables and 50 clauses, seed 0
- approximately 5,000 SDD nodes
- 47 Klay layers
- 7,616 total sparse rows
- 11,462 total sparse edges
- log semiring
- gradients with respect to both positive and negative literal weights

Only the batch size changes between the small, medium, and large workloads.
This isolates batching behavior without introducing the unpredictable SDD
compilation growth caused by increasing the variable count.

## Methodology

The native JAX and SpringMV backends receive the same CSR topology and input
weights. Inputs remain resident on the GPU. Both forward and
forward-plus-backward functions are compiled with `jax.jit`; construction,
preprocessing, compilation, random input generation, and host-to-device
transfer are excluded from the steady-state timings.

The forward-plus-backward measurement uses `jax.value_and_grad` around the
complete Klay evaluation. Each timing is synchronized with
`jax.block_until_ready`.

## Environment

- GPU: NVIDIA GeForce RTX 4090, 24 GB
- NVIDIA driver: 580.126.16
- CUDA toolkit used to build SpringMV: 13.0
- JAX and jaxlib: 0.12.1
- Klay commit: `bc3e2ff`
- SpringMV commit: `923175b`

## Interpretation

SpringMV's forward path is already beneficial for a single example. Its
custom backward kernels have more fixed overhead at batch size 1, so native
JAX remains about 15% faster for the smallest combined workload. Fused
batched forward and backward kernels amortize that overhead: SpringMV is 9%
faster at batch size 128 and 62% faster at batch size 1024.

For Klay workloads that evaluate or train on meaningful batches, these
results support using SpringMV as the JAX sparse backend. Native JAX remains a
reasonable choice for single-example workloads where backward performance
dominates.

## Reproduction

From the Klay repository, using the SpringMV virtual environment:

```bash
for batch in 1 128 1024; do
  ../SpringMV/.venv/bin/python -m benchmark.benchmark_wmc \
    --benchmark sdd \
    --target jax \
    --jax-backend both \
    --semiring log \
    --device cuda:0 \
    --nb_vars 100 \
    --batch_size "$batch" \
    --warmup 10 \
    --iterations 50
done
```

The benchmark caches JSON results under `results/` and skips an existing
configuration. Move or remove the corresponding cached file before requesting
a fresh run.

## Limitations

This is an initial comparison on one GPU, one generated circuit, one random
seed, and the log semiring. It measures steady-state compiled execution rather
than preprocessing or compilation. Additional circuit structures, GPUs, and
batch sizes are needed before treating the speedups as general performance
guarantees.
