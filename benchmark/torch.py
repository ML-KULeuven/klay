from time import perf_counter

import torch
from pysdd.iterator import SddIterator

from .utils import numpy_weights


def _torch_weights(nb_vars: int, semiring: str, device: str, batch_size: int):
    weights, neg_weights = numpy_weights(nb_vars, semiring, batch_size)
    weights = torch.as_tensor(weights).to(device)
    neg_weights = torch.as_tensor(neg_weights).to(device)
    weights.requires_grad = True
    neg_weights.requires_grad = True
    return weights, neg_weights


def benchmark_klay_torch(circuit, nb_vars, semiring, nb_repeats=10, device='cpu', batch_size=None):
    results = {}
    t1 = perf_counter()
    circuit_forward = circuit.to_torch_module(semiring, compile=False).to(device)
    results['to_torch'] = perf_counter() - t1

    results['sparsity'] = circuit_forward.sparsity(nb_vars)

    t1 = perf_counter()
    circuit_forward = torch.compile(circuit_forward, mode="reduce-overhead")
    results['jit compile'] = perf_counter() - t1

    timings = []
    with torch.no_grad():
        for _ in range(nb_repeats + 2):  # 2 warmup runs
            weights, neg_weights = _torch_weights(nb_vars, semiring, device, batch_size=batch_size)
            t1 = perf_counter()
            circuit_forward(weights, neg_weights)
            if device == 'cuda':
                torch.cuda.synchronize()
            timings.append(perf_counter() - t1)
    results['forward (cold)'] = timings[0]
    results['forward (warm)'] = timings[2:]

    timings = []
    for _ in range(nb_repeats + 2):
        weights, neg_weights = _torch_weights(nb_vars, semiring, device, batch_size=batch_size)
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
            weights, neg_weights = _torch_weights(nb_vars, 'log',  device, batch_size=batch_size)
            t1 = perf_counter()
            _eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_forward.append(perf_counter() - t1)

    t_backward = []
    for _ in range(nb_repeats + 2):
        weights, neg_weights = _torch_weights(nb_vars, 'log', device, batch_size=batch_size)
        t1 = perf_counter()
        _eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device).mean().backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        t_backward.append(perf_counter() - t1)
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def _eval_sdd_torch_naive(manager, sdd, pos_weights, neg_weights, device):
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

    return iterator.depth_first(sdd, _formula_evaluator)
