from time import perf_counter

import jax

from .utils import numpy_weights


def _jax_weights(nb_vars: int, semiring: str, batch_size: int):
    weights, neg_weights = numpy_weights(nb_vars, semiring, batch_size)
    return jax.numpy.array(weights), jax.numpy.array(neg_weights)


def _to_dot_graphs(func, *args):
    with open("unopt.dot", "w") as f:
        x = jax.xla_computation(func)(*args)
        f.write(x.as_hlo_dot_graph())
    with open("opt.dot", "w") as f:
        x = func.lower(*args).compile()
        print(x.cost_analysis())
        x = jax.lib.xla_client._xla.hlo_module_from_text(x)
        x = jax.lib.xla_client._xla.hlo_module_to_dot_graph(x)
        f.write(x.as_text())


def benchmark_klay_jax(circuit, nb_vars, semiring, nb_repeats=10, device='cpu', batch_size=None):
    results = {}
    device_id = int(device.split(":")[1]) if ":" in device else 0
    device_name = device.split(":")[0]
    device = jax.devices(device_name)[device_id]

    with jax.default_device(device):
        t1 = perf_counter()
        _circuit_forward = circuit.to_jax_function(semiring)
        results["to_jax"] = perf_counter() - t1

        circuit_forward2 = lambda x, y: _circuit_forward(x, y)[0]
        if batch_size is not None:
            circuit_forward_vmap = jax.vmap(circuit_forward2)
            circuit_forward = lambda x, y: jax.numpy.mean(circuit_forward_vmap(x, y))
        else:
            circuit_forward = circuit_forward2
        circuit_forward = jax.jit(circuit_forward)

        timings = []
        for _ in range(nb_repeats+2):  # 2 warmup runs
            weights, neg_weights = _jax_weights(nb_vars, semiring, batch_size=batch_size)
            t1 = perf_counter()
            circuit_forward(weights, neg_weights).block_until_ready()
            timings.append(perf_counter() - t1)
        results['forward (cold)'] = timings[0]
        results['forward (warm)'] = timings[2:]

        circuit_backward = jax.jit(jax.value_and_grad(circuit_forward, argnums=(0, 1)))
        timings = []
        for _ in range(nb_repeats+2):
            weights, neg_weights = _jax_weights(nb_vars, semiring, batch_size=batch_size)
            t1 = perf_counter()
            v, grad = circuit_backward(weights, neg_weights)
            jax.block_until_ready((v, grad))
            timings.append(perf_counter() - t1)
        results[' +backward (cold)'] = timings[0]
        results[' +backward (warm)'] = timings[2:]
    return results
