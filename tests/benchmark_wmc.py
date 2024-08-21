from time import time


import jax.numpy as jnp
import torch
import numpy as np

import klay
from klay.utils import generate_random_sdd, pysdd_wmc


def benchmark_jax(circuit, weights, nb_repeats=5):
    weights = jnp.log(jnp.array(weights))
    circuit_func = circuit.to_jax_function()
    timings = []
    for _ in range(nb_repeats):
        t1 = time()
        circuit_func(weights)[0]
        timings.append(time() - t1)
    return np.mean(timings[2:])


def benchmark_torch(circuit, weights, nb_repeats=5):
    weights = torch.as_tensor(weights).log()
    module = circuit.to_torch_module()
    timings = []
    for _ in range(nb_repeats):
        t1 = time()
        module(weights).item()
        timings.append(time() - t1)
    return np.mean(timings[2:])


def run_benchmark(nb_vars=72, seed=1):
    sdd, weights = generate_random_sdd(nb_vars, nb_vars//2, seed=seed)
    print('Nb Nodes:', sdd.count())
    circuit = klay.Circuit()
    t1 = time()
    circuit.add_sdd(sdd)
    print(f'Time to layerize: {time() - t1:.4f}s')
    delta_jax = benchmark_jax(circuit, weights)
    print(f"time jax: {delta_jax:.4f}s")
    delta_torch = benchmark_torch(circuit, weights)
    print(f"time torch: {delta_torch:.4f}s")


if __name__ == "__main__":
    run_benchmark()