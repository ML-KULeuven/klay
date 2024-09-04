import json
import os
from pathlib import Path
import argparse

import numpy as np

import klay
from klay.utils import generate_random_dimacs, benchmark_klay_jax, benchmark_klay_torch, benchmark_pysdd
from klay.compile import compile_sdd


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
            results.update(benchmark_klay_jax(circuit, weights, device=device))
        elif target == "torch":
            results.update(benchmark_klay_torch(circuit, weights, device=device))
    return results


def run_d4_bench(file_name: str, target:str, device: str):
    weights = [np.random.rand() for _ in range(1000)]  # hacky, but should be enough
    circuit = klay.Circuit()

    circuit.add_D4_from_file(file_name)
    results = {"klay_nodes": circuit.nb_nodes()}
    print("nb nodes", circuit.nb_nodes())
    if target == "jax":
        results.update(benchmark_klay_jax(circuit, weights, device=device))
    elif target == "torch":
        results.update(benchmark_klay_torch(circuit, weights, device=device))
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
