import argparse
import logging
import time

import klay
import numpy as np
from benchmark.jax import benchmark_klay_jax
from benchmark.torch import benchmark_klay_torch
from pysdd.sdd import SddManager, Vtree


CIRCUITS = ["sudoku_4", "4-grid", "seq_fun", "warcraft_12"]

log = logging.getLogger(__name__)


def print_results(results):
    width = max(len(k) for k in results)
    for k, v in results.items():
        v = np.array(v)
        if v.size > 1:
            log.info(f"  {k:{width}}  {v.mean():.3g} ± {v.std():.3g}")
        else:
            log.info(f"  {k:{width}}  {v:.3g}")


def main(backend, batch_size, device):
    file_name = f"experiments/nesy/{backend}_{device.split(':')[0]}_b{batch_size}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(file_name, mode="w"),
        ],
    )

    for name in CIRCUITS:
        log.info(f"\n### Running {name} (batch size={batch_size}, device={device}) ###")
        sdd_file = f"experiments/nesy/circuits/{name}.sdd"
        vtree_file = f"experiments/nesy/circuits/{name}.vtree"

        vtree = Vtree.from_file(vtree_file.encode())
        manager = SddManager.from_vtree(vtree)
        sdd = manager.read_sdd_file(sdd_file.encode())
        log.info(f"Loaded SDD with {sdd.count() + sdd.size()} nodes.")

        t1 = time.perf_counter()
        circuit = klay.Circuit()
        circuit.add_sdd_from_file(sdd_file)
        delta = time.perf_counter() - t1
        log.info(f"Layerized in {circuit.nb_nodes()} nodes and {len(circuit.to_torch_module().layers)} layers")
        log.info(f"  in {delta:2g}s.")

        if backend == "torch":
            log.info(f"Benchmarking Torch")
            result = benchmark_klay_torch(circuit, 1000, 'log', device=device, batch_size=batch_size)
            print_results(result)

        elif backend == "jax":
            log.info(f"Benchmarking Jax")
            results = benchmark_klay_jax(circuit, 1000, 'log', device=device, batch_size=batch_size)
            print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default="128")
    parser.add_argument('-d', '--device', type=str, default="cpu")
    parser.add_argument('-e', '--backend', type=str, default='torch', choices=["jax", "torch"])
    args = parser.parse_args()

    main(args.backend, args.batch_size, args.device)
