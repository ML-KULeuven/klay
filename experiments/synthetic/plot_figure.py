import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-paper')

textwidth = 5.50107
aspect_ratio = 3/8
scale = 1
width = textwidth * scale
height = width * aspect_ratio

def load_timings(results):
    for folder_name in list(results.keys()):
        folder = Path('results') / folder_name
        if not folder.exists():
            continue
        print("Loading", folder)

        for experiment in folder.iterdir():
            if os.path.getsize(experiment) == 0:
                continue
            with open(experiment) as f:
                data = json.load(f)
                data_point = np.mean(data['backward']) * 1000
                results[folder_name].append(data_point)

    for k, v in results.items():
        v.sort()


def load_stat(results, folder_name, name):
    folder = Path('results') / folder_name
    if not folder.exists():
        return
    print("Loading", name, 'from', folder)

    results[name] = []
    for experiment in folder.iterdir():
        if os.path.getsize(experiment) == 0:
            continue
        assert experiment.suffix == ".txt", f"File {experiment} is not a .txt file"
        with open(experiment) as f:
            data = json.load(f)
            results[name].append(data[name])
    results[name].sort()


def plot_sdd_stats():
    results = dict()
    load_stat(results, "sdd_torch_log_cuda", "klay_nodes")
    load_stat(results, "sdd_torch_log_cuda", "sdd_nodes")
    load_stat(results, "sdd_torch_log_cuda", "sparsity")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(width, height), constrained_layout=True)

    ax1.plot(results['sdd_nodes'], label="Nb of Nodes in SDD", color='black')
    ax1.plot(results['klay_nodes'], label="Nb of Nodes in KLay", color='black', linestyle='--')

    ax1.set_ylabel("Nb of Nodes")
    legend = ax1.legend(fancybox=False)
    legend.get_frame().set_linewidth(0.)

    ax2.plot(list(reversed(results['sparsity'])), color='black')
    ax2.set_ylabel("Sparsity")

    for ax in [ax1, ax2]:
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel("Instances")
        ax.set_xlim(0, len(results["sdd_nodes"])-1)

    fig.savefig("sdd_stats.pdf", bbox_inches='tight')


def plot_sdd():
    results = {
        "sdd_jax_log_cpu": [], "sdd_jax_log_cuda": [],
        "sdd_torch_log_cpu": [], "sdd_torch_log_cuda": [],
        "sdd_pysdd_log_cpu": [],
        "sdd_torch_real_cpu": [], "sdd_torch_real_cuda": [],
        "sdd_juice_cpu": [], "sdd_juice_cuda": [],
        "sdd_pysdd_real_cpu": [],
    }
    load_timings(results)
    #

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(width, height), constrained_layout=True)

    ### Left figure
    timings = np.cumsum(results['sdd_torch_log_cpu'])
    ax1.plot(timings, label="KLay (torch, cpu)", color='red')
    timings = np.cumsum(results['sdd_torch_log_cuda'])
    ax1.plot(timings, label="KLay (torch, cuda)", color='red', linestyle='--')
    timings = np.cumsum(results['sdd_jax_log_cpu'])
    ax1.plot(timings, label="KLay (jax, cpu)", color='blue')
    timings = np.cumsum(results['sdd_jax_log_cuda'])
    ax1.plot(timings, label="KLay (jax, cuda)", color='blue', linestyle='--')

    timings = np.cumsum(results['sdd_pysdd_log_cpu'])
    ax1.plot(timings, label="Post-order (cpu)", color='black')

    ax1.set_ylabel("Cumulative Time (ms)")
    ax1.set_ylim(0.02, 50000)
    ax1.set_title("Log Semiring")


    ### Middle figure
    timings = np.cumsum(results['sdd_torch_real_cpu'])
    ax2.plot(timings, color='red')
    timings = np.cumsum(results['sdd_torch_real_cuda'])
    ax2.plot(timings, color='red', linestyle='--')

    timings = np.cumsum(results['sdd_juice_cpu'])
    ax2.plot(timings, label="Juice (cpu)", color='green')
    timings = np.cumsum(results['sdd_juice_cuda'])
    ax2.plot(timings, label="Juice (cuda)", color='green', linestyle="--")

    timings = np.cumsum(results['sdd_pysdd_real_cpu'])
    ax2.plot(timings, color='black')

    ax2.set_ylim(0.02, 50000)
    ax2.set_title("Real Semiring")

    for ax in [ax1, ax2]:
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel("Instances")
        ax.set_xlim(0, len(results["sdd_torch_log_cpu"])-1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, fancybox=False, edgecolor="black", loc='center', bbox_to_anchor=(1.14, 0.6))
    legend.get_frame().set_linewidth(0.)

    fig.savefig("sdd_bench.pdf", bbox_inches='tight')



def plot_d4():
    results = {"d4_jax_log_cpu": [], "d4_jax_log_cuda": [], "d4_torch_log_cpu": [], "d4_torch_log_cuda": [], "d4_kompyle": []}
    load_timings(results)
    load_stat(results, "d4_jax_log_cpu", "d4_nodes")
    load_stat(results, "d4_jax_log_cpu", "klay_nodes")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(width, height), constrained_layout=True)

    timings = np.cumsum(sorted(results['d4_torch_log_cpu']))
    ax2.plot(timings, label="KLay (torch, cpu)", color='red')
    timings = np.cumsum(sorted(results['d4_torch_log_cuda']))
    ax2.plot(timings, label="KLay (torch, cuda)", color='red', linestyle='--')
    timings = np.cumsum(sorted(results['d4_jax_log_cpu']))
    ax2.plot(timings, label="KLay (jax, cpu)", color='blue')
    timings = np.cumsum(sorted(results['d4_jax_log_cuda']))
    ax2.plot(timings, label="KLay (jax, cuda)", color='blue', linestyle='--')
    timings = np.cumsum(sorted(results['d4_kompyle']))
    ax2.plot(timings, label="Post-order (cpu)", color='black')

    ax1.set_ylabel("Cumulative Time (ms)")

    ax1.plot(results['d4_nodes'], label="Nb of Nodes in d-DNNF",  color='black')
    ax1.plot(results['klay_nodes'], label="Nb of Nodes in KLay", color='black', linestyle='--')
    legend = ax1.legend(fancybox=False)
    legend.get_frame().set_linewidth(0.)

    ax1.set_ylabel("Nb of Nodes")

    for ax in [ax1, ax2]:
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel("Instances")
        ax.set_xlim(0, len(results["d4_torch_log_cpu"])-1)

    lines, labels = ax2.get_legend_handles_labels()
    legend = fig.legend(lines, labels, fancybox=False, edgecolor="black", loc='center', bbox_to_anchor=(1.14, 0.7))
    legend.get_frame().set_linewidth(0.)


    fig.savefig("d4_bench.pdf", bbox_inches='tight')


if __name__ == "__main__":
    plot_sdd()
    plot_sdd_stats()
    plot_d4()
