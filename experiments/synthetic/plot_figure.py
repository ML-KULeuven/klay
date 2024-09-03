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

        for experiment in folder.iterdir():
            assert experiment.suffix == ".txt", f"File {experiment} is not a .txt file"
            with open(experiment) as f:
                data = json.load(f)
                data_point = data['sdd_edges'], np.mean(data['backward']) * 1000
                results[folder_name].append(data_point)

    plt.figure(figsize=(5, 3))
    for name, data in results.items():
        if data:
            nb_nodes, timings = zip(*data)
            plt.scatter(nb_nodes, timings, label=name[4:].replace("_", " ").title(), s=4)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Edges")
    plt.ylabel("Time (ms)")
    plt.xlim(10**3, 10**9)
    # plt.title("SDD Backpropagation Time")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("sdd_backpropagation_time.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
