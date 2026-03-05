import argparse
import time

import klay
import numpy as np
from klay.utils import benchmark_klay_torch, benchmark_klay_jax
from pysdd.sdd import SddManager, Vtree


CIRCUITS = ["sudoku_4", "4-grid", "seq_fun", "warcraft_12"]


def print_results(results):
    for k, v in results.items():
        v = np.array(v)
        if v.size > 1:
            print(f"  {k}:\t {v.mean():.2g} ± {v.std():.2g}")
        else:
            print(f"  {k}:\t {v:.2g}")


def main(batch_size):
    for name in CIRCUITS:
        print(f"### Running {name} (bs={batch_size}) ###")
        sdd_file = f"experiments/nesy/circuits/{name}.sdd"
        vtree_file = f"experiments/nesy/circuits/{name}.vtree"

        vtree = Vtree.from_file(vtree_file.encode())
        manager = SddManager.from_vtree(vtree)
        sdd = manager.read_sdd_file(sdd_file.encode())
        print(f"Loaded SDD with {sdd.count() + sdd.size()} nodes.")

        t1 = time.perf_counter()
        circuit = klay.Circuit()
        circuit.add_sdd_from_file(sdd_file)
        delta = time.perf_counter() - t1
        print(f"Layerized in {circuit.nb_nodes()} nodes and {len(circuit.to_torch_module().layers)} layers")
        print(f"  in {delta:2g}s.")

        for device in ['cpu']:
            print(f"Benchmarking Torch {device}")
            result = benchmark_klay_torch(circuit, 1000, 'log', device=device, batch_size=batch_size)
            print_results(result)

            print(f"Benchmarking Jax  {device}")
            results = benchmark_klay_jax(circuit, 1000, 'log', device=device, batch_size=batch_size)
            print_results(results)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=None)
    args = parser.parse_args()

    main(args.batch_size)
