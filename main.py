import math
import random
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
    klay.brr("test.sdd")

    s = Source.from_file("tensorized.dot")
    s.view()
    s = Source.from_file("layerized.dot")
    s.view()


def test_with_pysdd(nb_vars: int, verbose=True, repeats=1):
    manager, sdd = generate_random_sdd(nb_vars, nb_vars//2)
    weights = torch.empty(nb_vars, dtype=torch.float32)
    weights.uniform_(0, 1)
    weights = weights.log()
    weights.requires_grad = True
    ground_truth = wmc_pysdd(manager, sdd, weights, verbose)

    t1 = time()
    circuit = klay.Circuit.from_SDD_file("test.sdd")
    if verbose:
        print(f"KLayerization in {time()-t1:.2f}s")
    t1 = time()
    kl = circuit.to_layered_module()
    if verbose:
        print(f"KTensorization in {time()-t1:.2f}s")
    for i in range(repeats):
        t1 = time()
        result = kl(weights)
        if verbose:
            print(f"KLayer forward in {time()-t1:.4f}s")
            print(f"PySDD\t{ground_truth:.7f}")
            print(f'KLAY\t{result.item():.7f}')

        t1 = time()
        result.backward()
        if verbose:
            print(f"KLayer backward in {time()-t1:.4f}s")
        weights.grad.zero_()

    if verbose:
        print()
    return ground_truth, result


def fuzz_tester(nb_vars: int, nb_tests=1000):
    for _ in tqdm(range(nb_tests)):
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
    print("Generated SDD", manager.count())
    return manager, sdd


def log1mexp(x):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )


def wmc_pysdd(manager, sdd, weights, verbose=True):
    wmc = sdd.wmc(log_mode=True)
    for i, v in enumerate(manager.vars):
        wmc.set_literal_weight(v, weights[i])
        wmc.set_literal_weight(~v, log1mexp(weights[i]))
    t1 = time()
    result = wmc.propagate()
    if verbose:
        print(f"PySDD WMC in {time()-t1:.4f}s")
    return result


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(53)
    # fuzz_tester(30)
    # for i in range(10):
    test_with_pysdd(66)
    # s = Source.from_file("tensorized.dot")
    # s.view()
    # s = Source.from_file("layerized.dot")
    # s.view()

