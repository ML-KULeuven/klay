import pytest

import torch

import klay
from klay.utils import generate_random_sdd, dict_product, pysdd_wmc


@pytest.fixture(params=dict_product({
    'seed': list(range(10)),
    'var_count': [6],
    'clause_count': [3],
    'clause_length': [3],
}))
def sdd(request):
    yield generate_random_sdd(**request.param)


def test_sdd_wmc(sdd):
    sdd, weights = sdd
    wmc_gt = pysdd_wmc(sdd, weights)

    klay_weights = torch.tensor(weights).log()
    circuit = klay.Circuit()
    circuit.add_sdd(sdd)
    kl = circuit.to_torch_module()
    result = float(kl(klay_weights).item())
    assert wmc_gt == pytest.approx(result, abs=1e-5)
