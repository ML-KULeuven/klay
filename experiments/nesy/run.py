import random

import klay
import numpy as np
from klay.utils import benchmark_klay_jax, benchmark_klay_torch, benchmark_sdd_torch_naive
from pysdd.sdd import SddManager, Vtree


CIRCUITS = ["4-grid", "road_r", "sudoku_4", "warcraft_12"]


def print_results(results):
    for k, v in results.items():
        v = np.array(v) * 1000
        print(f"{k}: {np.mean(v):.2f} \pm {np.std(v):.2f}")


def main():
    for name in CIRCUITS:
        print("### Running", name)
        sdd_file = f"experiments/nesy/circuits/{name}.sdd"
        vtree_file = f"experiments/nesy/circuits/{name}.vtree"

        vtree = Vtree.from_file(vtree_file.encode())
        manager = SddManager.from_vtree(vtree)
        sdd = manager.read_sdd_file(sdd_file.encode())
        print("Loaded SDD", sdd.count())

        circuit = klay.Circuit()
        circuit.add_SDD_from_file(sdd_file)
        print("Layerized in nodes", circuit.nb_nodes())
        weights = [random.random() for _ in range(1000)]  # hacky, but should be enough

        for device in ['cpu', 'cuda']:
            print(f"Benchmarking Torch {device}")
            result = benchmark_klay_torch(circuit, weights, device=device)
            print_results(result)

            # print(f"Benchmarking Jax {device}")
            # result = benchmark_klay_jax(circuit, weights, device=device)
            # print_results(result)

            print(f"Benchmarking Torch Naive {device}")
            result = benchmark_sdd_torch_naive(manager, sdd, weights, device=device)
            print_results(result)


if __name__ == "__main__":
    main()
