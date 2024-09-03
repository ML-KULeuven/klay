import json
import math
import os
from pathlib import Path
from time import perf_counter as time
from array import array
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import torch

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
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


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
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def benchmark_pysdd(sdd, weights, nb_repeats=10, device='cpu'):
    assert device == 'cpu'
    # WARNING: pysdd computes both the forward and backward passes in propagate
    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = sdd.wmc(log_mode=True)
    wmc_manager.set_literal_weights_from_array(pysdd_weights)

    timings = []
    for _ in range(nb_repeats+2):
        t1 = time()
        wmc_manager.propagate()
        timings.append(time() - t1)
    return {'backward': timings[2:]}


def run_sdd_bench(nb_vars: int, target: str, seed: int, device: str = 'cpu'):
    generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=seed)
    sdd = compile_sdd('tmp.cnf')
    nb_nodes = sdd.count()
    print(f"Nb of Nodes in SDD: {nb_nodes//1000}k")
    weights = [np.random.rand() for _ in range(nb_vars)]
    results = {'sdd_nodes': nb_nodes, "sdd_edges": get_edge_count(sdd)}

    if target == 'pysdd':
        results.update(benchmark_pysdd(sdd, weights, device=device))
    else:
        circuit = klay.Circuit()
        circuit.add_sdd(sdd)
        results['klay_nodes'] = circuit.nb_nodes()
        if target == "jax":
            results.update(benchmark_jax(circuit, weights, device=device))
        elif target == "torch":
            results.update(benchmark_torch(circuit, weights, device=device))
    return results


def run_d4_bench(file_name: str, target:str, device: str):
    weights = [np.random.rand() for _ in range(1000)]  # hacky, but should be enough
    circuit = klay.Circuit()

    circuit.add_D4_from_file(file_name)
    results = {"klay_nodes": circuit.nb_nodes()}
    print("nb nodes", circuit.nb_nodes())
    if target == "jax":
        results.update(benchmark_jax(circuit, weights, device=device))
    elif target == "torch":
        results.update(benchmark_torch(circuit, weights, device=device))
    return results


def get_edge_count(sdd):
    sdd.save(bytes(Path("tmp.sdd")))
    count = 0
    with open("tmp.sdd", "rb") as f:
        for line in f:
            line = line.decode()
            if line[0] == 'D':
                count += 1
                count += int(line.split()[2]) * 2
    print(f"Nb of Edges in SDD: {count}")
    os.remove("tmp.sdd")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--nb_vars', nargs="+", type=int)
    parser.add_argument('-r', '--nb_repeats', type=int, default=1)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-t', '--target', default='jax')
    parser.add_argument('-b', '--benchmark', required=True)
    args = parser.parse_args()

    for nb_vars in args.nb_vars:
        print(f'Benchmarking {args.benchmark}-{args.target} on {args.device}  ({nb_vars} variables)')
        for seed in range(args.nb_repeats):
            if args.benchmark == 'sdd':
                results = run_sdd_bench(nb_vars, target=args.target, device=args.device, seed=seed)
            if args.benchmark == 'd4':
                results = run_d4_bench('experiments/synthetic/d4_large.nnf', target=args.target, device=args.device)

            file_name = f"results/{args.benchmark}_{args.target}_{args.device}/v{nb_vars}_{seed}.txt"
            Path(file_name).parent.mkdir(exist_ok=True, parents=True)
            with open(file_name, 'w') as f:
                json.dump(results, f)


if __name__ == "__main__":
    main()
