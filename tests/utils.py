import math
import random
from array import array

import torch
from pysdd.iterator import SddIterator

from klay.torch.utils import log1mexp


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


def eval_pysdd(sdd: "SddNode", weights: list[float]):
    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = sdd.wmc(log_mode=True)
    wmc_manager.set_literal_weights_from_array(pysdd_weights)
    return wmc_manager.propagate()


def eval_d4_torch_naive(nnf_file: str, weights: list[float], neg_weights: list[float] = None):
    with open(nnf_file) as f:
        nnf_string = f.read()

    weights = torch.as_tensor(weights, dtype=torch.float32)
    if neg_weights is None:
        neg_weights = log1mexp(weights)
    else:
        neg_weights = torch.as_tensor(neg_weights, dtype=torch.float32)
    weights = torch.stack([neg_weights, weights], dim=1)

    one = torch.tensor(0., dtype=torch.float32, device=weights.device)
    zero = torch.tensor(float('-inf'), dtype=torch.float32, device=weights.device)

    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in lines:
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([zero, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([one, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            if len(literals) == 0:
                lits_val = nodes[target][0]
            else:
                ix1 = [abs(lit) - 1 for lit in literals]
                ix2 = [int(lit > 0) for lit in literals]
                lit_weights = weights[..., ix1, ix2]
                lits_val = nodes[target][0] + lit_weights.sum(dim=-1)

            if nodes[source][1] == 'o':
                nodes[source][0] = torch.logaddexp(nodes[source][0], lits_val)
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] + lits_val
    return nodes[1][0]
