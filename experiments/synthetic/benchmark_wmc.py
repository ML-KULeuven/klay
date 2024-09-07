import json
from pathlib import Path
import argparse

import numpy as np

import klay
from klay.utils import generate_random_dimacs, benchmark_klay_jax, benchmark_klay_torch, benchmark_pysdd
from klay.compile import compile_sdd, compile_d4


def run_sdd_bench(nb_vars: int, target: str, seed: int, device: str = 'cpu'):
    generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=seed)
    sdd = compile_sdd('tmp.cnf')
    nb_nodes = sdd.count() + sdd.size()
    print(f"Nb of Nodes in SDD: {nb_nodes//1000}k")
    weights = [np.random.rand() for _ in range(nb_vars)]
    results = {'sdd_nodes': nb_nodes}

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


def run_d4_bench(nb_vars: int, target:str, seed: int, device: str):
    generate_random_dimacs('tmp.cnf', nb_vars, 2*nb_vars, seed=seed)
    compile_d4('tmp.cnf', 'tmp.nnf')
    weights = [np.random.rand() for _ in range(nb_vars)]
    circuit = klay.Circuit()
    circuit.add_D4_from_file('tmp.nnf')
    results = {"klay_nodes": circuit.nb_nodes(), 'd4_nodes': get_d4_node_count('tmp.nnf')}
    print("nb nodes", circuit.nb_nodes())
    if target == "jax":
        results.update(benchmark_klay_jax(circuit, weights, device=device))
    elif target == "torch":
        results.update(benchmark_klay_torch(circuit, weights, device=device))
    return results


def get_d4_node_count(nnf_file):
    with open(nnf_file) as f:
        for line in reversed(list(f)):
            if line[0] in ('a', 'o', 't', 'f'):
                return int(line.split(' ')[1])
    return None


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
                results = run_d4_bench(nb_vars, target=args.target, device=args.device, seed=seed)

            file_name = f"results/{args.benchmark}_{args.target}_{args.device}/v{nb_vars}_{seed}.txt"
            Path(file_name).parent.mkdir(exist_ok=True, parents=True)
            with open(file_name, 'w') as f:
                json.dump(results, f)


if __name__ == "__main__":
    main()
