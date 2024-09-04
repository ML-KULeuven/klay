import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    results = {"sdd_pysdd_cpu": [], "sdd_jax_cpu": [], "sdd_jax_cuda": [], "sdd_torch_cpu": [], "sdd_torch_cuda": []}
    for folder_name in results.keys():
        folder = Path('results') / folder_name
        print("Loading", folder)
        if not folder.exists():
            continue
        nb_nodes_klay = []
        nb_nodes_sdd = []

        for experiment in folder.iterdir():
            assert experiment.suffix == ".txt", f"File {experiment} is not a .txt file"
            with open(experiment) as f:
                data = json.load(f)
                data_point = np.mean(data['backward']) * 1000
                results[folder_name].append(data_point)
                if 'klay_nodes' in data:
                    nb_nodes_klay.append(data['klay_nodes'])
                nb_nodes_sdd.append(data['sdd_nodes'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    timings = np.cumsum(sorted(results['sdd_pysdd_cpu']))
    ax1.plot(timings, label="PySDD (cpu)", linewidth=1.5, color='black')
    timings = np.cumsum(sorted(results['sdd_jax_cpu']))
    ax1.plot(timings, label="KLay (Jax, cpu)", linewidth=1.5, color='blue')
    timings = np.cumsum(sorted(results['sdd_jax_cuda']))
    ax1.plot(timings, label="KLay (Jax, cuda)", linewidth=1.5, color='blue', linestyle='--')
    timings = np.cumsum(sorted(results['sdd_torch_cpu']))
    ax1.plot(timings, label="KLay (Torch, cpu)", linewidth=1.5, color='red')
    timings = np.cumsum(sorted(results['sdd_torch_cuda']))
    ax1.plot(timings, label="KLay (Torch, cuda)", linewidth=1.5, color='red', linestyle='--')

    ax1.set_ylabel("Cumulative Time (ms)")

    ax2.plot(sorted(nb_nodes_sdd), label="Nb of Nodes in SDD", linewidth=1.5, color='black')
    ax2.plot(sorted(nb_nodes_klay), label="Nb of Nodes after Layerization", linewidth=1.5, color='black', linestyle='--')

    ax2.set_ylabel("Nb of Nodes")

    for ax in [ax1, ax2]:
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel("Instances")
        ax.set_xlim(0, len(results["sdd_pysdd_cpu"]))
        ax.legend()

    fig.savefig("sdd_bench.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
