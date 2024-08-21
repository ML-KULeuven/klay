import pytest

import jax.numpy as jnp

import klay
from klay.utils import generate_random_sdd, dict_product, pysdd_wmc


@pytest.fixture(params=dict_product({
    'seed': list(range(100)),
    'var_count': [40],
    'clause_count': [20],
    'clause_length': [3],
}))
def sdd(request):
    yield generate_random_sdd(**request.param)


def test_sdd_wmc(sdd):
    sdd, weights = sdd
    wmc_gt = pysdd_wmc(sdd, weights)

    klay_weights = jnp.log(jnp.array(weights))
    circuit = klay.Circuit()
    circuit.add_sdd(sdd)
    circuit_func = circuit.to_jax_function()
    result = circuit_func(klay_weights)[0]
    assert wmc_gt == pytest.approx(result, abs=1e-5)
