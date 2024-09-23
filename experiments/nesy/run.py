import argparse

import klay
import numpy as np
from klay.utils import benchmark_klay_jax, benchmark_klay_torch, benchmark_sdd_torch_naive
from pysdd.sdd import SddManager, Vtree


CIRCUITS = ["sudoku_4", "4-grid", "seq_fun", "warcraft_12"]


def print_results(results):
    for k, v in results.items():
        v = np.array(v) * 1000  # convert to ms
        print(f"{k}: {np.mean(v):.2f} \pm {np.std(v):.2f}")


def main(batch_size):
    for name in CIRCUITS:
        print(f"### Running {name} (batch size {batch_size})")
        sdd_file = f"experiments/nesy/circuits/{name}.sdd"
        vtree_file = f"experiments/nesy/circuits/{name}.vtree"

        vtree = Vtree.from_file(vtree_file.encode())
        manager = SddManager.from_vtree(vtree)
        sdd = manager.read_sdd_file(sdd_file.encode())
        print(f"Loaded SDD {sdd.count() + sdd.size()}")

        circuit = klay.Circuit()
        circuit.add_SDD_from_file(sdd_file)
        print(f"Layerized in {circuit.nb_nodes()} nodes and {len(circuit.to_torch_module().layers)} layers")

        for device in ['cpu', 'cuda']:
            print(f"Benchmarking Torch {device}")
            result = benchmark_klay_torch(circuit, 1000, 'log', device=device, batch_size=batch_size)
            print_results(result)

            print(f"Benchmarking Torch Naive {device}")
            result = benchmark_sdd_torch_naive(manager, sdd, 1000, device=device, batch_size=batch_size)
            print_results(result)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=None)
    args = parser.parse_args()

    main(args.batch_size)
