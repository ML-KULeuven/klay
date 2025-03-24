import random

import numpy as np
import pytest

import torch
from tqdm import tqdm

import klay
from klay.utils import generate_random_dimacs, eval_pysdd #, torch_wmc_d4
from klay.compile import compile_sdd#, compile_d4


def check_sdd(sdds, weights):
    wmc_gts = [eval_pysdd(sdd, weights) for sdd in sdds]

    weights = torch.tensor(weights).log()
    circuit = klay.Circuit()
    for sdd in sdds:
        circuit.add_sdd(sdd)
    kl = circuit.to_torch_module()
    result = kl(weights)
    for i, wmc_gt in enumerate(wmc_gts):
        assert wmc_gt == pytest.approx(float(result[i]), abs=1e-4), f"Expected {wmc_gt}, got {result}"

    kl = torch.vmap(kl)
    result_vmap = kl(weights.unsqueeze(0))
    assert np.allclose(result, result_vmap), f"Expected {result}, got {result_vmap}"



def check_d4(nnf_files, weights):
    weights = torch.tensor(weights)
    weights.requires_grad = True
    wmc_gts = [torch_wmc_d4(nnf_file, weights) for nnf_file in nnf_files]
    for wmc_gt in wmc_gts:
        wmc_gt.backward()
    grad_gt = weights.grad.numpy()
    weights.grad.zero_()
    wmc_gts = [float(wmc_gt.log()) for wmc_gt in wmc_gts]

    circuit = klay.Circuit()
    for nnf_file in nnf_files:
        circuit.add_d4_from_file(nnf_file)
    kl = circuit.to_torch_module()
    result = kl(weights.log())
    result.backward(torch.ones_like(result))
    grad = weights.grad.numpy()
    for i, wmc_gt in enumerate(wmc_gts):
        assert wmc_gt == pytest.approx(float(result[i]), abs=1e-4), f"Expected {wmc_gt}, got {result}"
    assert np.allclose(grad_gt, grad), f"Expected {grad_gt}, got {grad}"


def fuzzer_multi_rooted(nb_trials, nb_vars, nb_roots, seed_offset=None):
    if seed_offset is None:
        import sys
        seed_offset = random.randint(0, sys.maxsize)
        print("Random seed offset: ", seed_offset)
    for i in tqdm(range(nb_trials)):
        for j in range(nb_roots):
            generate_random_dimacs(f'tmp{j}.cnf', nb_vars, nb_vars//2, seed=i + (j+1)*seed_offset)
        weights = [random.random() for _ in range(nb_vars)]
        sdds = [compile_sdd(f'tmp{j}.cnf') for j in range(nb_roots)]
        check_sdd(sdds, weights)

        # for j in range(nb_roots):
        #     compile_d4(f'tmp{j}.cnf', f'tmp{j}.nnf')
        # nnf_files = [f"tmp{j}.nnf" for j in range(nb_roots)]
        # check_d4(nnf_files, weights)


if __name__ == "__main__":
    nb_trails = 100
    nb_vars = 25
    nb_roots = 5
    seed_offset = None
    print("Running Multi rooted Fuzz Tester on 3-CNFs")
    print("Number of Trials:", nb_trails)
    print("Number of Variables:", nb_vars)
    print("Number of Roots:", nb_roots)
    if seed_offset is not None:
        print("Seed offset: ", seed_offset)
    fuzzer_multi_rooted(nb_trails, nb_vars, nb_roots, seed_offset)
