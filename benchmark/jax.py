from time import perf_counter

import jax
import numpy as np

from klay.jax import create_knowledge_layer

from .utils import numpy_weights


def _jax_weights(nb_vars: int, semiring: str, batch_size: int | None):
    weights, neg_weights = numpy_weights(nb_vars, semiring, batch_size)
    return jax.numpy.asarray(weights), jax.numpy.asarray(neg_weights)


def _timings(function, arguments, warmup: int, iterations: int):
    timings = []
    for index in range(warmup + iterations):
        weights, neg_weights = arguments[index % len(arguments)]
        start = perf_counter()
        result = function(weights, neg_weights)
        jax.block_until_ready(result)
        elapsed = perf_counter() - start
        if index >= warmup:
            timings.append(elapsed)
    return timings


def _summarize(label: str, timings: list[float]):
    values = np.asarray(timings) * 1e3
    print(
        f"{label:<44} {np.median(values):9.4f} "
        f"[{np.min(values):9.4f}, {np.max(values):9.4f}] ms"
    )


def _benchmark_backend(
    indices,
    semiring: str,
    backend: str,
    arguments,
    batch_size: int | None,
    warmup: int,
    iterations: int,
):
    start = perf_counter()
    raw_forward = create_knowledge_layer(
        *indices,
        semiring=semiring,
        backend=backend,
    )
    construction_time = perf_counter() - start

    def single_forward(x, y):
        return raw_forward(x, y)[0]

    if batch_size is None:
        forward = single_forward
    else:
        batched_forward = jax.vmap(single_forward)

        def forward(x, y):
            return jax.numpy.mean(batched_forward(x, y))

    forward = jax.jit(forward)
    forward_backward = jax.jit(
        jax.value_and_grad(forward, argnums=(0, 1))
    )

    # Record compilation separately from steady-state execution.
    compile_start = perf_counter()
    jax.block_until_ready(forward(*arguments[0]))
    forward_compile_time = perf_counter() - compile_start

    compile_start = perf_counter()
    jax.block_until_ready(forward_backward(*arguments[0]))
    backward_compile_time = perf_counter() - compile_start

    forward_timings = _timings(forward, arguments, warmup, iterations)
    forward_backward_timings = _timings(
        forward_backward,
        arguments,
        warmup,
        iterations,
    )
    return {
        "construction": construction_time,
        "forward_compile": forward_compile_time,
        "forward_backward_compile": backward_compile_time,
        "forward": forward_timings,
        "forward_backward": forward_backward_timings,
    }


def benchmark_klay_jax(
    circuit,
    nb_vars,
    semiring,
    device="cpu",
    batch_size=None,
    backend="jax",
    warmup=5,
    iterations=20,
):
    """Benchmark one or both Klay JAX sparse-reduction backends."""
    if backend not in ("jax", "springmv", "both"):
        raise ValueError("backend must be 'jax', 'springmv', or 'both'")

    device_id = int(device.split(":")[1]) if ":" in device else 0
    device_name = device.split(":")[0]
    if device_name == "cuda":
        device_name = "gpu"
    selected_device = jax.devices(device_name)[device_id]

    with jax.default_device(selected_device):
        # _get_indices mutates the circuit by adding a root layer. Extract it
        # exactly once so both backends benchmark the same CSR topology.
        indices = circuit._get_indices()
        column_indices, row_offsets = indices
        total_rows = sum(len(offsets) - 1 for offsets in row_offsets)
        total_edges = sum(len(columns) for columns in column_indices)
        print(
            f"layers={len(column_indices)} total-rows={total_rows} "
            f"total-edges={total_edges} batch={batch_size or 1}"
        )
        # Reuse resident buffers: timing measures the compiled computation,
        # not NumPy generation or host-to-device transfer.
        arguments = (_jax_weights(nb_vars, semiring, batch_size),)
        jax.block_until_ready(arguments)

        backends = ("jax", "springmv") if backend == "both" else (backend,)
        results = {}
        for current_backend in backends:
            current = _benchmark_backend(
                indices,
                semiring,
                current_backend,
                arguments,
                batch_size,
                warmup,
                iterations,
            )
            results[current_backend] = current
            display_name = (
                "native JAX" if current_backend == "jax" else "SpringMV"
            )
            _summarize(
                f"{display_name} forward",
                current["forward"],
            )
            _summarize(
                f"{display_name} forward + backward",
                current["forward_backward"],
            )

        if backend == "both":
            print("ratios are native JAX median / SpringMV median")
            for operation in ("forward", "forward_backward"):
                native = np.median(results["jax"][operation])
                springmv = np.median(results["springmv"][operation])
                print(f"{operation:<44} {native / springmv:9.2f}x")
            return results
        return results[backend]
