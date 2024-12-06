import random

import numpy as np
import pytest

import torch
from tqdm import tqdm

import klay
from klay.utils import generate_random_dimacs, eval_pysdd, eval_d4_torch_naive
from klay.compile import compile_sdd# , compile_d4


def check_sdd(sdd, weights):
    wmc_gt = eval_pysdd(sdd, weights)

    weights = torch.tensor(weights).log()
    circuit = klay.Circuit()
    circuit.add_sdd(sdd)
    kl = circuit.to_torch_module()
    result = float(kl(weights))
    assert wmc_gt == pytest.approx(result, abs=1e-4), f"Expected {wmc_gt}, got {result}"

    kl = torch.vmap(kl)
    result_vmap = kl(weights.unsqueeze(0))
    assert np.allclose(result, result_vmap), f"Expected {result}, got {result_vmap}"


def check_d4(nnf_file, weights):
    weights = torch.tensor(weights).log()
    weights.requires_grad = True
    wmc_gt = eval_d4_torch_naive(nnf_file, weights)
    wmc_gt.backward()
    grad_gt = weights.grad.numpy()
    weights.grad.zero_()
    wmc_gt = float(wmc_gt)

    circuit = klay.Circuit()
    circuit.add_D4_from_file(nnf_file)
    kl = circuit.to_torch_module()
    result = kl(weights)
    result.backward()
    grad = weights.grad.numpy()
    assert wmc_gt == pytest.approx(float(result), abs=1e-4), f"Expected {wmc_gt}, got {result}"
    assert np.allclose(grad_gt, grad), f"Expected {grad_gt}, got {grad}"


def fuzzer(nb_trials, nb_vars):
    import sys
    seed_offset = random.randint(0, sys.maxsize)
    print("Seed offset: ", seed_offset)
    for i in tqdm(range(nb_trials)):
        generate_random_dimacs('tmp.cnf', nb_vars, nb_vars//2, seed=i + seed_offset)
        weights = [random.random() for _ in range(nb_vars)]

        sdd = compile_sdd('tmp.cnf')
        check_sdd(sdd, weights)

        # compile_d4('tmp.cnf', 'tmp.nnf')
        # check_d4("tmp.nnf", weights)


if __name__ == "__main__":
    nb_trails = 50
    nb_vars = 50
    print("Running Fuzz Tester on 3-CNFs")
    print("Number of Trials:", nb_trails)
    print("Number of Variables:", nb_vars)
    fuzzer(nb_trails, nb_vars)
