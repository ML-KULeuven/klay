from time import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import klay
from klay.utils import generate_random_dimacs
from klay.compile import compile_sdd


def benchmark_jax(circuit, weights, nb_repeats=5):
    weights = jnp.log(jnp.array(weights))
    circuit_forward = circuit.to_jax_function()
    timings_forward = []
    for _ in range(nb_repeats):
        t1 = time()
        circuit_forward(weights)
        timings_forward.append(time() - t1)

    circuit_backward = jax.jit(jax.grad(circuit_forward))
    timings_backward = []
    for _ in range(nb_repeats):
        t1 = time()
        circuit_backward(weights)
        timings_backward.append(time() - t1)

    forward_timings.append(np.mean(timings_forward[2:]))
    backward_timings.append(np.mean(timings_backward[2:]))


def run_sdd_bench(nb_vars: int):
    for seed in tqdm(range(nb_repeats)):
        generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=seed)
        sdd = compile_sdd('tmp.cnf')
        weights = [np.random.rand() for _ in range(nb_vars)]
        circuit = klay.Circuit()
        circuit.nb_vars = nb_vars

        t1 = time()
        circuit.add_sdd(sdd)
        layerize_timings.append(time() - t1)
        benchmark_jax(circuit, weights)


def run_d4_bench(file_name: str):
    weights = [np.random.rand() for _ in range(nb_vars)]
    circuit = klay.Circuit()
    circuit.nb_vars = nb_vars

    t1 = time()
    circuit.add_D4_from_file(file_name)
    layerize_timings.append(time() - t1)
    benchmark_jax(circuit, weights)


if __name__ == "__main__":
    nb_vars = 50
    nb_repeats = 10

    forward_timings = []
    backward_timings = []
    layerize_timings = []

    # run_sdd_bench(nb_vars, seed)
    run_d4_bench('tests/d4_medium.nnf')

    print(f'Forward Timings: {np.mean(forward_timings):.5f}s')
    print(f'Backward Timings: {np.mean(backward_timings):.5f}s')
    print(f'Layerize Timings: {np.mean(layerize_timings):.3f}s')
