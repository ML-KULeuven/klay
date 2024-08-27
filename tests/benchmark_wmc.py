import math
from time import perf_counter as time
from array import array
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

import klay
from klay.utils import generate_random_dimacs
from klay.compile import compile_sdd


def benchmark_jax(circuit, weights, nb_repeats=10, device='cpu'):
    with jax.default_device(jax.devices(device)[0]):
        weights = jnp.log(jnp.array(weights))
        _circuit_forward = circuit.to_jax_function()
        circuit_forward = lambda x: _circuit_forward(x)[0]
        t_forward = []
        for _ in range(nb_repeats+2): # 2 warmup runs
            t1 = time()
            circuit_forward(weights).block_until_ready()
            t_forward.append(time() - t1)

        circuit_backward = jax.jit(jax.value_and_grad(circuit_forward))
        t_backward = []
        for _ in range(nb_repeats+2):
            t1 = time()
            v, grad = circuit_backward(weights)
            v.block_until_ready()
            t_backward.append(time() - t1)

    forward_timings.append(t_forward[2:])
    backward_timings.append(t_backward[2:])


def benchmark_torch(circuit, weights, nb_repeats=10, device='cpu'):
    weights = torch.as_tensor(weights).log().to(device)
    circuit_forward = circuit.to_torch_module().to(device)
    t_forward = []
    with torch.inference_mode():
        for _ in range(nb_repeats+2):
            t1 = time()
            circuit_forward(weights)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_forward.append(time() - t1)

    t_backward = []
    weights = weights.detach()
    weights.requires_grad = True
    for _ in range(nb_repeats + 2):
        t1 = time()
        circuit_forward(weights).backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        t_backward.append(time() - t1)
        weights.grad.zero_()

    forward_timings.append(t_forward[2:])
    backward_timings.append(t_backward[2:])


def benchmark_pysdd(sdd, weights, nb_repeats=5, device='cpu'):
    assert device == 'cpu'
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
    backward_timings.append(timings[2:])


def run_sdd_bench(nb_vars: int, target: str, device: str = 'cpu'):
    sdd_nodes = []
    for seed in tqdm(range(args.nb_repeats)):
        generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=seed)
        sdd = compile_sdd('tmp.cnf')
        sdd_nodes.append(sdd.count())
        weights = [np.random.rand() for _ in range(nb_vars)]

        if target == 'pysdd':
            benchmark_pysdd(sdd, weights, device=device)
            continue

        circuit = klay.Circuit()
        t1 = time()
        circuit.add_sdd(sdd)
        layerize_timings.append(time() - t1)
        if target == "jax":
            benchmark_jax(circuit, weights, device=device)
        elif target == "torch":
            benchmark_torch(circuit, weights, device=device)
    print(f"Nb of Nodes in SDD: {np.mean(sdd_nodes):.2f} ± {np.std(sdd_nodes):.2f}")


def run_d4_bench(file_name: str, target:str, device: str):
    weights = [np.random.rand() for _ in range(1000)]  # hacky, but should be enough
    circuit = klay.Circuit()

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

    print(f'Benchmarking {args.benchmark}-{args.target} on {args.device}  ({args.nb_vars} variables)')

    forward_timings = []
    backward_timings = []
    layerize_timings = []

    if args.benchmark == 'sdd':
        run_sdd_bench(args.nb_vars, target=args.target, device=args.device)
    if args.benchmark == 'd4':
        run_d4_bench('tests/d4_large.nnf', target=args.target, device=args.device)

    if forward_timings:
        mean_timings = [np.mean(runs) for runs in zip(*forward_timings)]
        mean_timings = np.array(mean_timings) * 1000  # in milliseconds
        print(f'Forward Timings: {mean_timings.mean():.4f} ± {mean_timings.std():.4f}')
    if backward_timings:
        mean_timings = [np.mean(runs) for runs in zip(*backward_timings)]
        mean_timings = np.array(mean_timings) * 1000  # in milliseconds
        print(f'Backward Timings: {mean_timings.mean():.4f} ± {mean_timings.std():.4f}')
    if layerize_timings:
        print(f'Layerize Timings: {np.mean(layerize_timings):.3f} ± {np.std(layerize_timings):.3f}')  # in seconds