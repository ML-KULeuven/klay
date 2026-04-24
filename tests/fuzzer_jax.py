import random
import pytest

pytest.importorskip("jax")
pytest.importorskip("pysdd")

import jax.numpy as jnp
from tqdm import tqdm
from pysdd.sdd import SddManager

import klay
from .utils import generate_random_dimacs, eval_pysdd


def check_sdd_jax(sdd, weights):
    wmc_gt = eval_pysdd(sdd, weights)

    klay_weights = jnp.log(jnp.array(weights))
    circuit = klay.Circuit()
    circuit.add_sdd(sdd)
    kl = circuit.to_jax_function()
    result = float(kl(klay_weights).item())
    assert wmc_gt == pytest.approx(result, abs=1e-4), f"Expected {wmc_gt}, got {result}"


def fuzzer(nb_trials, nb_vars):
    for i in tqdm(range(nb_trials)):
        generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=i)
        weights = [random.random() for _ in range(nb_vars)]

        sdd = SddManager.from_cnf_file(b'tmp.cnf')[1]
        check_sdd_jax(sdd, weights)


def test_sdd():
    fuzzer(10, 20)


if __name__ == "__main__":
    nb_trails = 50
    nb_vars = 50
    print("Running Fuzz Tester on 3-CNFs")
    print("Number of Trials:", nb_trails)
    print("Number of Variables:", nb_vars)
    fuzzer(nb_trails, nb_vars)
