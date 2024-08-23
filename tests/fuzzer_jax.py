import random

import pytest

from tqdm import tqdm
import jax.numpy as jnp

import klay
from klay.utils import generate_random_dimacs, pysdd_wmc, torch_wmc_d4
from klay.compile import compile_sdd, compile_d4


def check_sdd_jax(sdd, weights):
    wmc_gt = pysdd_wmc(sdd, weights)

    klay_weights = jnp.log(jnp.array(weights))
    circuit = klay.Circuit()
    circuit.nb_vars = len(weights)
    circuit.add_sdd(sdd)
    kl = circuit.to_jax_function()
    result = float(kl(klay_weights).item())
    assert wmc_gt == pytest.approx(result, abs=1e-4), f"Expected {wmc_gt}, got {result}"


def check_d4_jax(nnf_file, weights):
    wmc_gt = torch_wmc_d4(nnf_file, weights)

    klay_weights = jnp.log(jnp.array(weights))
    circuit = klay.Circuit()
    circuit.nb_vars = len(weights)
    circuit.add_D4_from_file(nnf_file)
    kl = circuit.to_jax_function()
    result = float(kl(klay_weights).item())
    assert wmc_gt == pytest.approx(result, abs=1e-4), f"Expected {wmc_gt}, got {result}"


def fuzzer(nb_trials, nb_vars):
    for i in tqdm(range(nb_trials)):
        generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=i)
        weights = [random.random() for _ in range(nb_vars)]

        sdd = compile_sdd('tmp.cnf')
        check_sdd_jax(sdd, weights)

        compile_d4('tmp.cnf', 'tmp.nnf')
        check_d4_jax("tmp.nnf", weights)


if __name__ == "__main__":
    nb_trails = 50
    nb_vars = 50
    print("Running Fuzz Tester on 3-CNFs")
    print("Number of Trials:", nb_trails)
    print("Number of Variables:", nb_vars)
    fuzzer(nb_trails, nb_vars)
