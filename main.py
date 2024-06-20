from time import time
from functools import reduce
from pathlib import Path
from random import randint, choice

from graphviz import Source
from pysdd.sdd import SddManager, Vtree
import torch
from tqdm import tqdm

import klay


def main():
    klay.brr(name="test.sdd")

    s = Source.from_file("tensorized.dot")
    s.view()
    s = Source.from_file("layerized.dot")
    s.view()


def test_with_pysdd(nb_vars: int, verbose=True):
    manager, sdd = generate_random_sdd(nb_vars, nb_vars//2)
    weights = torch.empty(nb_vars, dtype=torch.float32)
    weights.uniform_(0, 1)
    weights.round_(decimals=2)
    ground_truth = wmc_pysdd(manager, sdd, weights)

    t1 = time()
    klay.brr(name="test.sdd")
    if verbose:
        print(f"KLayerization in {time()-t1:.2f}s")
    kl = klay.torch_backend.KnowledgeLayer(klay.parse_tensors("tensors.txt"))
    t1 = time()
    result = kl(weights).item()
    if verbose:
        print(f"KLayer forward in {time()-t1:.2f}s")
        print("RESULTS:")
        print(f"PySDD\t{ground_truth:.7f}")
        print(f'KLAY\t{result:.7f}')
    return ground_truth, result


def fuzz_tester(nb_vars: int):
    for _ in tqdm(range(1000)):
        gt, pred = test_with_pysdd(nb_vars, False)
        assert abs(gt - pred) < 1e-5, f"Error: {gt} != {pred}"
    print("All tests passed!")


def generate_random_sdd(nb_vars: int, nb_clauses: int, clause_length: int = 3):
    vtree = Vtree(var_count=nb_vars, vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    sdd = manager.true()
    for _ in range(nb_clauses):
        lits = [randint(1, nb_vars) * choice([1, -1]) for _ in range(clause_length)]
        lits = [manager.l(lit) for lit in lits]
        sdd &= reduce(manager.disjoin, lits)

    sdd.save(bytes(Path("test.sdd")))
    # print("Generated SDD", manager.count())
    return manager, sdd


def wmc_pysdd(manager, sdd, weights):
    wmc = sdd.wmc(log_mode=False)
    for i, v in enumerate(manager.vars):
        wmc.set_literal_weight(v, weights[i])
        wmc.set_literal_weight(~v, 1-weights[i])
    return wmc.propagate()


if __name__ == "__main__":
    fuzz_tester(25)
    # s = Source.from_file("tensorized.dot")
    # s.view()
    # s = Source.from_file("layerized.dot")
    # s.view()

