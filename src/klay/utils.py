import math
import random
from array import array

import torch


def generate_random_dimacs(file_name: str, var_count: int, clause_count: int, seed: int = 1, clause_length: int = 3):
    """
    Generate a random k-CNF formula and save it to a file in DIMACS format.
    """
    random.seed(seed)

    with open(file_name, "w") as f:
        f.write(f"p cnf {var_count} {clause_count}\n")
        for _ in range(clause_count):
            clause = [random.randint(1, var_count) * random.choice([1, -1])
                        for _ in range(clause_length)]
            f.write(" ".join(map(str, clause)) + " 0\n")


def pysdd_wmc(sdd: "SddNode", weights: list[float]):
    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = sdd.wmc(log_mode=True)
    wmc_manager.set_literal_weights_from_array(pysdd_weights)
    return wmc_manager.propagate()


def torch_wmc_d4(nnf_file: str, weights: list[float]):
    with open(nnf_file) as f:
        nnf_string = f.read()

    ONE = torch.tensor(1., dtype=torch.float32)
    ZERO = torch.tensor(0., dtype=torch.float32)

    weights = torch.as_tensor(weights, dtype=torch.float32)
    weights = torch.stack([1 - weights, weights], dim=1)

    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in lines:
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([ZERO, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([ONE, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            if len(literals) == 0:
                lits_val = nodes[target][0]
            else:
                ix1 = [abs(lit) - 1 for lit in literals]
                ix2 = [int(lit > 0) for lit in literals]
                lit_weights = weights[ix1, ix2]
                lits_val = nodes[target][0] * lit_weights.prod()

            if nodes[source][1] == 'o':
                nodes[source][0] = nodes[source][0] + lits_val
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] * lits_val
    return nodes[1][0]


def plot_circuit_overhead(module):
    layer_widths = []
    layer_edges = []
    for layer in module.layers:
        layer_width = layer.csr.shape[0] - 1
        layer_widths.append(layer_width)
        layer_edges.append(layer.ptrs.shape[0])

    xx = list(range(len(layer_widths)))
    import matplotlib.pyplot as plt
    plt.plot(layer_widths)
    plt.plot(layer_edges)
    plt.fill_between(xx, layer_widths, alpha=0.2, label="overhead")
    plt.fill_between(xx, layer_widths, layer_edges, alpha=0.2, label="useful computation")
    plt.legend(["width", "edges"])
    plt.title("Layer utilization")
    # plt.yscale("log")
    plt.xlabel("Layer")
    plt.show()
