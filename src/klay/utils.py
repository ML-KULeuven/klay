import math
import random
import itertools
from array import array
from functools import reduce

from pysdd.sdd import SddManager, Vtree, SddNode


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def generate_random_sdd(var_count: int, clause_count: int, seed: int = 1, clause_length: int = 3):
    """
    Generate a random k-CNF formula and compile it into an SDD, along with random weights.
    """
    random.seed(seed)
    vtree = Vtree(var_count=var_count, vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    sdd = manager.true()
    for _ in range(clause_count):
        lits = [random.randint(1, var_count) * random.choice([1, -1]) for _ in
                range(clause_length)]
        lits = [manager.l(lit) for lit in lits]
        sdd &= reduce(manager.disjoin, lits)

    weights = [random.random() for _ in range(var_count)]
    return sdd, weights


def pysdd_wmc(sdd: SddNode, weights: list[float]):
    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = sdd.wmc(log_mode=True)
    wmc_manager.set_literal_weights_from_array(pysdd_weights)
    return wmc_manager.propagate()
