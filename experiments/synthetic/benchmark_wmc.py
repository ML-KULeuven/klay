import json
from pathlib import Path
import argparse

import numpy as np

import klay
from klay.utils import generate_random_dimacs, benchmark_klay_jax, benchmark_klay_torch, benchmark_pysdd
from klay.compile import compile_sdd, compile_d4


def run_sdd_bench(nb_vars: int, target: str, semiring: str, seed: int, device: str = 'cpu'):
    generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=seed)
    sdd = compile_sdd('tmp.cnf')
    nb_nodes = sdd.count() + sdd.size()
    print(f"Nb of Nodes in SDD: {nb_nodes//1000}k")
    results = {'sdd_nodes': nb_nodes}

    # save sdd and vtree for juice
    sdd.save(bytes(Path(f'results/sdd/v{nb_vars}_{seed}.sdd')))
    sdd.vtree().save(bytes(Path(f'results/sdd/v{nb_vars}_{seed}.vtree')))

    if target == 'pysdd':
        results.update(benchmark_pysdd(sdd, nb_vars, semiring, device=device))
    else:
        circuit = klay.Circuit()
        circuit.add_sdd(sdd)
        results['klay_nodes'] = circuit.nb_nodes()
        if target == "jax":
            results.update(benchmark_klay_jax(circuit, nb_vars, semiring, device=device))
        elif target == "torch":
            results.update(benchmark_klay_torch(circuit, nb_vars, semiring, device=device))
        else:
            raise ValueError(f"Unknown target {target}")
    return results


def run_d4_bench(nb_vars: int, target:str, semiring: str, seed: int, device: str):
    generate_random_dimacs('tmp.cnf', nb_vars, 2*nb_vars, seed=seed)
    compile_d4('tmp.cnf', 'tmp.nnf')
    circuit = klay.Circuit()
    circuit.add_D4_from_file('tmp.nnf')
    results = {"klay_nodes": circuit.nb_nodes(), 'd4_nodes': get_d4_node_count('tmp.nnf')}
    print(f"Nb of Nodes in KLay: {circuit.nb_nodes()//1000}k")
    if target == "jax":
        results.update(benchmark_klay_jax(circuit, nb_vars, semiring, device=device))
    elif target == "torch":
        results.update(benchmark_klay_torch(circuit, nb_vars, semiring, device=device))
    else:
        raise ValueError(f"Unknown target {target}")
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
    parser.add_argument('-b', '--benchmark', required=True, choices=['sdd', 'd4'])
    parser.add_argument('-s', '--semiring', default='log', choices=['log', 'real'])
    args = parser.parse_args()

    for nb_vars in args.nb_vars:
        print(f'Benchmarking {args.benchmark}-{args.target} on {args.device}  ({nb_vars} variables)')
        for seed in range(args.nb_repeats):
            file_name = Path(f"results/{args.benchmark}_{args.target}_{args.semiring}_{args.device}/v{nb_vars}_{seed}.txt")
            if file_name.exists():
                continue
            if args.benchmark == 'sdd':
                results = run_sdd_bench(nb_vars, args.target, args.semiring, seed, args.device)
            if args.benchmark == 'd4':
                results = run_d4_bench(nb_vars, args.target, args.semiring, seed, args.device)

            file_name.parent.mkdir(exist_ok=True, parents=True)
            with open(file_name, 'w') as f:
                json.dump(results, f)


if __name__ == "__main__":
    main()
