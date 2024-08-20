from pathlib import Path
import pytest
from pysdd.sdd import SddManager, Vtree
import random
from functools import reduce
import itertools
from array import array
import math
import torch
import klay


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


@pytest.fixture(params=dict_product({
    'seed': list(range(1)),
    'var_count': [1, 2, 3],
    'clause_count': [1, 2, 3],
    'clause_length': [1, 2, 3],
}))
def sdd(request):
    random.seed(request.param['seed'])
    vtree = Vtree(var_count=request.param['var_count'], vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    sdd = manager.true()
    for _ in range(request.param['clause_count']):
        lits = [random.randint(1, request.param['var_count']) * random.choice([1, -1]) for _ in
                range(request.param['clause_length'])]
        lits = [manager.l(lit) for lit in lits]
        sdd &= reduce(manager.disjoin, lits)

    weights = [random.random() for _ in range(request.param['var_count'])]
    yield manager, sdd, weights


def test_sdd_wmc(sdd):
    manager, node, weights = sdd
    node.save(bytes(Path("test.sdd")))

    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = node.wmc()
    wmc_manager.set_literal_weights_from_array(pysdd_weights)
    wmc_gt = wmc_manager.propagate()

    klay_weights = torch.Tensor(weights).log()
    circuit = klay.Circuit()
    circuit.add_SDD_from_file("test.sdd")
    kl = circuit.to_layered_module()
    result = float(kl(klay_weights).item())
    assert wmc_gt == pytest.approx(result)
