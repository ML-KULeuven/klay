import math
import random
from time import time

from graphviz import Source
from pysdd.iterator import SddIterator
import torch
from tqdm import tqdm

import klay
from klay.utils import generate_random_sdd, pysdd_wmc


def test_with_pysdd(nb_vars: int, verbose=True, repeats=3):
    sdd, weights = generate_random_sdd(nb_vars, nb_vars//2)
    ground_truth = pysdd_wmc(sdd, weights)
    weights = torch.tensor(weights, requires_grad=True).log()

    t1 = time()
    circuit = klay.Circuit()
    circuit.add_sdd(sdd)
    if verbose:
        print(f"KLayerization in {time()-t1:.2f}s")
    t1 = time()
    kl = circuit.to_torch_module()
    if verbose:
        print(f"KTensorization in {time()-t1:.2f}s")
    for i in range(repeats):
        t1 = time()
        result = kl(weights)
        if verbose:
            print(f"KLayer forward in {time()-t1:.4f}s")
            print(f"PySDD\t{ground_truth:.7f}")
            print(f'KLAY\t{result.item():.7f}')
    return ground_truth, result


def fuzz_tester(nb_vars: int, nb_tests=1000):
    for _ in tqdm(range(nb_tests)):
        gt, pred = test_with_pysdd(nb_vars, False, repeats=1)
        assert abs(gt - pred) < 1e-5, f"Error: {gt} != {pred}"
    print("All tests passed!")


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


def wmc_backward(manager, sdd, weights, verbose=True):
    pos_neg_weights = torch.stack([log1mexp(weights), weights], dim=1)
    result = eval_sdd(manager, sdd, pos_neg_weights)
    t1 = time()
    result.backward()
    if verbose:
        print(f"PyTorch backward in {time()-t1:.4f}s")
    grad = weights.grad.clone()
    weights.grad.zero_()
    return grad


def eval_sdd(manager, sdd, weights) -> torch.Tensor:
    iterator = SddIterator(manager, smooth=False)

    def _formula_evaluator(node, r_values, *_):
        if node is not None:
            if node.is_literal():
                return weights[abs(node.literal) - 1, int(node.literal > 0)]
            elif node.is_true():
                return torch.tensor(0.0)
            elif node.is_false():
                return torch.tensor(float("-inf"))
        # Decision node
        return torch.logsumexp(torch.stack([v[0] + v[1] for v in r_values]), dim=0)

    result = iterator.depth_first(sdd, _formula_evaluator)
    return result


def test_backward(nb_vars: int):
    _, sdd, weights = klay.utils.generate_random_sdd(nb_vars, nb_vars//2)
    # ground_truth = wmc_backward(manager, sdd, weights, True)
    # print("Ground truth", ground_truth)
    circuit = klay.Circuit()
    circuit.add_sdd(sdd)
    print("Circuit", circuit.nb_nodes())
    kl = circuit.to_torch_module()
    result = kl(weights)
    t1 = time()
    result.backward()
    print(f"KLAY backward in {time()-t1:.4f}s")
    print("KLAY result", weights.grad)


def test_multirooted(nb_vars: int):
    _, sdd1, weights = generate_random_sdd(nb_vars, nb_vars//2)
    circuit = klay.Circuit()
    circuit.add_SDD(sdd1)
    _, sdd2, _ = generate_random_sdd(nb_vars//2, nb_vars//4)
    circuit.add_SDD_from_file(sdd2)
    kl = circuit.to_torch_module()
    result = kl(weights)
    print(result)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(52)
    test_with_pysdd(40)
    fuzz_tester(40)
    # for i in range(10):
    # test_multirooted(50)
    # s = Source.from_file("circuit.dot")
    # s.view()
    # s = Source.from_file("layerized.dot")
    # s.view()

