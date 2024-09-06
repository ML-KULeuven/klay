import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(results):
    for folder_name in list(results.keys()):
        folder = Path('results') / folder_name
        if not folder.exists():
            continue
        print("Loading", folder)
        for node_count in ('klay_nodes', 'sdd_nodes', 'd4_nodes'):
            results[node_count] = []

        for experiment in folder.iterdir():
            assert experiment.suffix == ".txt", f"File {experiment} is not a .txt file"
            with open(experiment) as f:
                data = json.load(f)
                data_point = np.mean(data['backward']) * 1000
                results[folder_name].append(data_point)
                for node_count in ('klay_nodes', 'sdd_nodes', 'd4_nodes'):
                    if node_count in data:
                        results[node_count].append(data[node_count])

    for k, v in results.items():
        v.sort()


def plot_sdd():
    results = {"sdd_pysdd_cpu": [], "sdd_jax_cpu": [], "sdd_jax_cuda": [], "sdd_torch_cpu": [], "sdd_torch_cuda": []}
    load_data(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    timings = np.cumsum(results['sdd_torch_cpu'])
    ax1.plot(timings, label="KLay (torch, cpu)", linewidth=1.5, color='red')
    timings = np.cumsum(results['sdd_torch_cuda'])
    ax1.plot(timings, label="KLay (torch, cuda)", linewidth=1.5, color='red', linestyle='--')
    timings = np.cumsum(results['sdd_jax_cpu'])
    ax1.plot(timings, label="KLay (jax, cpu)", linewidth=1.5, color='blue')
    timings = np.cumsum(results['sdd_jax_cuda'])
    ax1.plot(timings, label="KLay (jax, cuda)", linewidth=1.5, color='blue', linestyle='--')
    timings = np.cumsum(results['sdd_pysdd_cpu'])
    ax1.plot(timings, label="PySDD (cpu)", linewidth=1.5, color='black')

    ax1.set_ylabel("Cumulative Time (ms)")

    ax2.plot(results['sdd_nodes'], label="Nb of Nodes in SDD", linewidth=1.5, color='black')
    ax2.plot(results['klay_nodes'], label="Nb of Nodes after Layerization", linewidth=1.5, color='black', linestyle='--')

    ax2.set_ylabel("Nb of Nodes")

    for ax in [ax1, ax2]:
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel("Instances")
        ax.set_xlim(0, len(results["sdd_pysdd_cpu"])-1)
        ax.legend()

    fig.savefig("sdd_bench.pdf", bbox_inches='tight')


def plot_d4():
    results = {"d4_jax_cpu": [], "d4_jax_cuda": [], "d4_torch_cpu": [], "d4_torch_cuda": []}
    load_data(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    timings = np.cumsum(sorted(results['d4_torch_cpu']))
    ax1.plot(timings, label="KLay (torch, cpu)", linewidth=1.5, color='red')
    timings = np.cumsum(sorted(results['d4_torch_cuda']))
    ax1.plot(timings, label="KLay (torch, cuda)", linewidth=1.5, color='red', linestyle='--')
    timings = np.cumsum(sorted(results['d4_jax_cpu']))
    ax1.plot(timings, label="KLay (jax, cpu)", linewidth=1.5, color='blue')
    timings = np.cumsum(sorted(results['d4_jax_cuda']))
    ax1.plot(timings, label="KLay (jax, cuda)", linewidth=1.5, color='blue', linestyle='--')

    ax1.set_ylabel("Cumulative Time (ms)")

    ax2.plot(results['d4_nodes'], label="Nb of Nodes in d-DNNF", linewidth=1.5, color='black')
    ax2.plot(results['klay_nodes'], label="Nb of Nodes after Layerization", linewidth=1.5, color='black', linestyle='--')

    ax2.set_ylabel("Nb of Nodes")

    for ax in [ax1, ax2]:
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel("Instances")
        ax.set_xlim(0, len(results["d4_jax_cpu"])-1)
        ax.legend()

    fig.savefig("d4_bench.pdf", bbox_inches='tight')


if __name__ == "__main__":
    plot_sdd()
    plot_d4()
