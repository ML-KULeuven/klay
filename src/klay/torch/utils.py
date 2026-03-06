import math
from time import perf_counter

import torch
from pysdd.iterator import SddIterator

from ..utils import numpy_weights

CUTOFF = -math.log(2)


def log1mexp(x, eps=1e-12):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    return torch.where(
        mask,
        (-x.clamp(min=CUTOFF).expm1() + eps).log(),
        (-x.clamp(max=CUTOFF).exp() + eps).log1p()
    )


def negate_real(x, eps):
    return 1 - x


def unroll_ixs(ixs):
    deltas = torch.diff(ixs)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=ixs.device)
    return ixs.repeat_interleave(repeats=deltas)


def eval_d4_torch_naive(nnf_file: str, weights: list[float], neg_weights: list[float] = None):
    with open(nnf_file) as f:
        nnf_string = f.read()

    weights = torch.as_tensor(weights, dtype=torch.float32)
    if neg_weights is None:
        neg_weights = log1mexp(weights)
    else:
        neg_weights = torch.as_tensor(neg_weights, dtype=torch.float32)
    weights = torch.stack([neg_weights, weights], dim=1)

    one = torch.tensor(0., dtype=torch.float32, device=weights.device)
    zero = torch.tensor(float('-inf'), dtype=torch.float32, device=weights.device)

    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in lines:
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([zero, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([one, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            if len(literals) == 0:
                lits_val = nodes[target][0]
            else:
                ix1 = [abs(lit) - 1 for lit in literals]
                ix2 = [int(lit > 0) for lit in literals]
                lit_weights = weights[..., ix1, ix2]
                lits_val = nodes[target][0] + lit_weights.sum(dim=-1)

            if nodes[source][1] == 'o':
                nodes[source][0] = torch.logaddexp(nodes[source][0], lits_val)
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] + lits_val
    return nodes[1][0]


def benchmark_klay_torch(circuit, nb_vars, semiring, nb_repeats=10, device='cpu', batch_size=None):
    results = {}
    t1 = perf_counter()
    circuit_forward = circuit.to_torch_module(semiring).to(device)
    results['to_torch'] = perf_counter() - t1

    results['sparsity'] = circuit_forward.sparsity(nb_vars)
    if batch_size is not None:
        circuit_forward = torch.vmap(circuit_forward)

    t1 = perf_counter()
    circuit_forward = torch.compile(circuit_forward, mode="reduce-overhead")
    results['jit compile'] = perf_counter() - t1

    timings = []
    with torch.no_grad():
        for _ in range(nb_repeats + 2):  # 2 warmup runs
            weights, neg_weights = torch_weights(nb_vars, semiring, device, batch_size=batch_size)
            t1 = perf_counter()
            circuit_forward(weights, neg_weights)
            if device == 'cuda':
                torch.cuda.synchronize()
            timings.append(perf_counter() - t1)
    results['forward (cold)'] = timings[0]
    results['forward (warm)'] = timings[2:]

    timings = []
    for _ in range(nb_repeats + 2):
        weights, neg_weights = torch_weights(nb_vars, semiring, device, batch_size=batch_size)
        t1 = perf_counter()
        circuit_forward(weights, neg_weights).mean().backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        timings.append(perf_counter() - t1)
    results[' +backward (cold)'] = timings[0]
    results[' +backward (warm)'] = timings[2:]
    return results


def benchmark_sdd_torch_naive(manager, sdd, nb_vars, nb_repeats=10, device='cpu', batch_size=None):
    t_forward = []
    with torch.inference_mode():
        for _ in range(nb_repeats+2):
            weights, neg_weights = torch_weights(nb_vars, 'log',  device, batch_size=batch_size)
            t1 = perf_counter()
            eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_forward.append(perf_counter() - t1)

    t_backward = []
    for _ in range(nb_repeats + 2):
        weights, neg_weights = torch_weights(nb_vars, 'log', device, batch_size=batch_size)
        t1 = perf_counter()
        eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device).mean().backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        t_backward.append(perf_counter() - t1)
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def eval_sdd_torch_naive(manager, sdd, pos_weights, neg_weights, device):
    iterator = SddIterator(manager, smooth=False)

    def _formula_evaluator(node, r_values, *_):
        if node is not None:
            if node.is_literal():
                literal = node.literal
                if literal < 0:
                    return neg_weights[..., -literal - 1]
                else:
                    return pos_weights[..., literal - 1]
            elif node.is_true():
                return torch.tensor(0., device=device)
            elif node.is_false():
                return torch.tensor(float('-inf'), device=device)
        # Decision node
        return torch.logsumexp(torch.stack([value[0] + value[1] for value in r_values]), dim=0)

    result = iterator.depth_first(sdd, _formula_evaluator)
    return result


def torch_weights(nb_vars: int, semiring: str, device: str, batch_size: int):
    weights, neg_weights = numpy_weights(nb_vars, semiring, batch_size)
    weights = torch.as_tensor(weights).to(device)
    neg_weights = torch.as_tensor(neg_weights).to(device)
    weights.requires_grad = True
    neg_weights.requires_grad = True
    return weights, neg_weights
