import math
from time import perf_counter
import random
from array import array

import torch
import jax


from klay.backends.torch_backend import log1mexp
from pysdd.iterator import SddIterator


def generate_random_dimacs(file_name: str, var_count: int, clause_count: int, seed: int = 1, clause_length: int = 3):
    """
    Generate a random k-CNF formula and save it to a file in DIMACS format.
    """
    random.seed(seed)

    with open(file_name, "w") as f:
        f.write(f"p cnf {var_count} {clause_count}\n")
        for _ in range(clause_count):
            clause = [random.randint(1, var_count) * random.choice([1, -1])
                        for _ in range(clause_length)]
            f.write(" ".join(map(str, clause)) + " 0\n")


def eval_pysdd(sdd: "SddNode", weights: list[float]):
    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = sdd.wmc(log_mode=True)
    wmc_manager.set_literal_weights_from_array(pysdd_weights)
    return wmc_manager.propagate()


def benchmark_pysdd(sdd, weights, nb_repeats=10, device='cpu'):
    assert device == 'cpu'
    # WARNING: pysdd computes both the forward and backward passes in propagate
    neg_weights = [1.0 - x for x in weights[::-1]]
    pysdd_weights = array('d', [math.log(x) for x in neg_weights + weights])
    wmc_manager = sdd.wmc(log_mode=True)
    wmc_manager.set_literal_weights_from_array(pysdd_weights)

    timings = []
    for _ in range(nb_repeats+2):
        t1 = perf_counter()
        wmc_manager.propagate()
        timings.append(perf_counter() - t1)
    return {'backward': timings[2:]}


def eval_d4_torch_naive(nnf_file: str, weights: list[float], neg_weights: list[float] = None):
    with open(nnf_file) as f:
        nnf_string = f.read()

    ONE = torch.tensor(0., dtype=torch.float32, device=weights.device)
    ZERO = torch.tensor(float('-inf'), dtype=torch.float32, device=weights.device)

    weights = torch.as_tensor(weights, dtype=torch.float32)
    if neg_weights is None:
        neg_weights = log1mexp(weights)
    else:
        neg_weights = torch.as_tensor(neg_weights, dtype=torch.float32)
    weights = torch.stack([neg_weights, weights], dim=1)

    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in lines:
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([ZERO, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([ONE, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            if len(literals) == 0:
                lits_val = nodes[target][0]
            else:
                ix1 = [abs(lit) - 1 for lit in literals]
                ix2 = [int(lit > 0) for lit in literals]
                lit_weights = weights[..., ix2, ix1]
                lits_val = nodes[target][0] + lit_weights.sum(axis=1)

            if nodes[source][1] == 'o':
                nodes[source][0] = torch.logaddexp(nodes[source][0], lits_val)
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] + lits_val
    return nodes[1][0]


def plot_circuit_overhead(module):
    layer_widths = []
    layer_edges = []
    for layer in module.layers:
        layer_width = layer.csr.shape[0] - 1
        layer_widths.append(layer_width)
        layer_edges.append(layer.ptrs.shape[0])

    xx = list(range(len(layer_widths)))
    import matplotlib.pyplot as plt
    plt.plot(layer_widths)
    plt.plot(layer_edges)
    plt.fill_between(xx, layer_widths, alpha=0.2, label="overhead")
    plt.fill_between(xx, layer_widths, layer_edges, alpha=0.2, label="useful computation")
    plt.legend(["width", "edges"])
    plt.title("Layer utilization")
    # plt.yscale("log")
    plt.xlabel("Layer")
    plt.show()


def benchmark_klay_jax(circuit, weights, nb_repeats=10, device='cpu'):
    with jax.default_device(jax.devices(device)[0]):
        weights = jax.numpy.log(jax.numpy.array(weights))
        _circuit_forward = circuit.to_jax_function()
        circuit_forward = lambda x: _circuit_forward(x)[0]
        t_forward = []
        for _ in range(nb_repeats+2): # 2 warmup runs
            t1 = perf_counter()
            circuit_forward(weights).block_until_ready()
            t_forward.append(perf_counter() - t1)

        circuit_backward = jax.jit(jax.value_and_grad(circuit_forward))
        t_backward = []
        for _ in range(nb_repeats+2):
            t1 = perf_counter()
            v, grad = circuit_backward(weights)
            v.block_until_ready()
            t_backward.append(perf_counter() - t1)
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def benchmark_klay_torch(circuit, weights, nb_repeats=10, device='cpu'):
    weights = torch.as_tensor(weights).log().to(device)
    circuit_forward = circuit.to_torch_module().to(device)
    t_forward = []
    with torch.inference_mode():
        for _ in range(nb_repeats+2):
            t1 = perf_counter()
            circuit_forward(weights)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_forward.append(perf_counter() - t1)

    t_backward = []
    weights = weights.detach()
    weights.requires_grad = True
    for _ in range(nb_repeats + 2):
        t1 = perf_counter()
        circuit_forward(weights).backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        t_backward.append(perf_counter() - t1)
        weights.grad.zero_()
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def benchmark_sdd_torch_naive(manager, sdd, weights, nb_repeats=10, device='cpu'):
    weights = torch.as_tensor(weights).log().to(device)
    neg_weights = log1mexp(weights)
    t_forward = []
    with torch.inference_mode():
        for _ in range(nb_repeats+2):
            t1 = perf_counter()
            eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_forward.append(perf_counter() - t1)

    t_backward = []
    weights = weights.detach()
    weights.requires_grad = True
    for _ in range(nb_repeats + 2):
        t1 = perf_counter()
        eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device).backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        t_backward.append(perf_counter() - t1)
        weights.grad.zero_()
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def eval_sdd_torch_naive(manager, sdd, pos_weights, neg_weights, device):
    iterator = SddIterator(manager, smooth=False)

    def _formula_evaluator(node, r_values, *_):
        if node is not None:
            if node.is_literal():
                literal = node.literal
                if literal < 0:
                    return neg_weights[-literal - 1]
                else:
                    return pos_weights[literal - 1]
            elif node.is_true():
                return torch.tensor(0., device=device)
            elif node.is_false():
                return torch.tensor(float('-inf'), device=device)
        # Decision node
        return torch.logsumexp(torch.stack([value[0] + value[1] for value in r_values]), dim=0)

    result = iterator.depth_first(sdd, _formula_evaluator)
    return result