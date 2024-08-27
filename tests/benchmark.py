import math
from time import time
from array import array
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

import klay
from klay.utils import generate_random_dimacs, plot_circuit_overhead
from klay.compile import compile_sdd


def benchmark_jax(circuit, weights, nb_repeats=5, device='cpu'):
    with jax.default_device(jax.devices(device)[0]):
        weights = jnp.log(jnp.array(weights))
        circuit_forward = circuit.to_jax_function()
        timings_forward = []
        for _ in range(nb_repeats):
            t1 = time()
            circuit_forward(weights).block_until_ready()
            timings_forward.append(time() - t1)

        circuit_backward = jax.jit(jax.value_and_grad(circuit_forward))
        timings_backward = []
        for _ in range(nb_repeats):
            t1 = time()
            v, grad = circuit_backward(weights)
            v.block_until_ready()
            timings_backward.append(time() - t1)

        forward_timings.append(np.mean(timings_forward[2:]))
        backward_timings.append(np.mean(timings_backward[2:]))


def benchmark_torch(circuit, weights, nb_repeats=5, device='cpu'):
    weights = torch.as_tensor(weights).log().to(device)
    circuit_forward = circuit.to_torch_module().to(device)
    plot_circuit_overhead(circuit_forward)
    timings_forward = []
    with torch.inference_mode():
        for _ in range(nb_repeats):
            t1 = time()
            circuit_forward(weights)
            if device == 'cuda':
                torch.cuda.synchronize()
            timings_forward.append(time() - t1)

    timings_backward = []
    weights = weights.detach()
    weights.requires_grad = True
    for _ in range(nb_repeats):
        t1 = time()
        circuit_forward(weights).backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        timings_backward.append(time() - t1)
        weights.grad.zero_()

    forward_timings.append(np.mean(timings_forward[2:]))
    backward_timings.append(np.mean(timings_backward[2:]))


def benchmark_pysdd(sdd, weights, nb_repeats=5, device='cpu'):
    assert device=='cpu'
    # WARNING: pysdd computes both the forward and backward passes in propagate
    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = sdd.wmc(log_mode=True)
    wmc_manager.set_literal_weights_from_array(pysdd_weights)

    timings = []
    for _ in range(nb_repeats):
        t1 = time()
        wmc_manager.propagate()
        timings.append(time() - t1)
    backward_timings.append(np.mean(timings[2:]))


def run_sdd_bench(nb_vars: int, target: str, device: str = 'cpu'):
    for seed in tqdm(range(args.nb_repeats)):
        generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=seed)
        sdd = compile_sdd('tmp.cnf')
        weights = [np.random.rand() for _ in range(nb_vars)]

        if target == 'pysdd':
            benchmark_pysdd(sdd, weights, device=device)
            continue

        circuit = klay.Circuit()
        circuit.nb_vars = nb_vars

        t1 = time()
        circuit.add_sdd(sdd)
        layerize_timings.append(time() - t1)
        if target == "jax":
            benchmark_jax(circuit, weights, device=device)
        elif target == "torch":
            benchmark_torch(circuit, weights, device=device)


def run_d4_bench(file_name: str, target:str, device: str):
    weights = [np.random.rand() for _ in range(args.nb_vars)]
    circuit = klay.Circuit()
    circuit.nb_vars = args.nb_vars

    t1 = time()
    circuit.add_D4_from_file(file_name)
    layerize_timings.append(time() - t1)
    print("nb nodes", circuit.nb_nodes())
    if target == "jax":
        benchmark_jax(circuit, weights, device=device)
    elif target == "torch":
        benchmark_torch(circuit, weights, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--nb_vars', type=int)
    parser.add_argument('-r', '--nb_repeats', type=int)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-t', '--target', default='jax')
    parser.add_argument('-b', '--benchmark', required=True)
    args = parser.parse_args()

    if args.benchmark == 'd4':
        args.nb_vars = 1000 # hacky, but should be enough
    print(f'Benchmarking {args.benchmark}-{args.target} on {args.device}  ({args.nb_vars} variables)')

    forward_timings = []
    backward_timings = []
    layerize_timings = []

    if args.benchmark == 'sdd':
        run_sdd_bench(args.nb_vars, target=args.target, device=args.device)
    if args.benchmark == 'd4':
        run_d4_bench('tests/d4_large.nnf', target=args.target, device=args.device)

    if forward_timings:
        print(f'Forward Timings: {1000*np.mean(forward_timings):.2f}ms')
    if backward_timings:
        print(f'Backward Timings: {1000*np.mean(backward_timings):.2f}ms')
    if layerize_timings:
        print(f'Layerize Timings: {np.mean(layerize_timings):.3f}s')
