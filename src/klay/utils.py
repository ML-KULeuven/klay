import math
from time import perf_counter
import random
from array import array

import torch
try:
    import jax
except ImportError:
    pass


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


def benchmark_pysdd(sdd, weights, semiring, nb_repeats=10, device='cpu'):
    assert device == 'cpu'
    # WARNING: pysdd computes both the forward and backward passes in propagate
    lit_weights = [1.0 - x for x in weights[::-1]] + weights
    if semiring == "log":
        lit_weights = [math.log(x) for x in lit_weights]
    pysdd_weights = array('d', lit_weights)
    wmc_manager = sdd.wmc(log_mode=(semiring == "log"))
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


def benchmark_klay_jax(circuit, nb_vars, semiring, nb_repeats=10, device='cpu'):
    with jax.default_device(jax.devices(device)[0]):
        _circuit_forward = circuit.to_jax_function(semiring)
        circuit_forward = lambda x: _circuit_forward(x)[0]
        t_forward = []
        for _ in range(nb_repeats+2): # 2 warmup runs
            weights, neg_weights = jax_weights(nb_vars, semiring)
            t1 = perf_counter()
            circuit_forward(weights).block_until_ready()
            t_forward.append(perf_counter() - t1)

        circuit_backward = jax.jit(jax.value_and_grad(circuit_forward))
        t_backward = []
        for _ in range(nb_repeats+2):
            weights, neg_weights = jax_weights(nb_vars, semiring)
            t1 = perf_counter()
            v, grad = circuit_backward(weights)
            grad.block_until_ready()
            t_backward.append(perf_counter() - t1)
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def benchmark_klay_torch(circuit, nb_vars, semiring, nb_repeats=10, device='cpu'):
    circuit_forward = circuit.to_torch_module(semiring).to(device)
    circuit_forward = torch.compile(circuit_forward, mode="reduce-overhead")

    t_forward = []
    with torch.no_grad():
        for _ in range(nb_repeats+2):
            weights, neg_weights = torch_weights(nb_vars, semiring,  device)
            t1 = perf_counter()
            circuit_forward(weights, neg_weights)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_forward.append(perf_counter() - t1)

    t_backward = []
    for _ in range(nb_repeats + 2):
        weights, neg_weights = torch_weights(nb_vars, semiring,  device)
        t1 = perf_counter()
        circuit_forward(weights, neg_weights).backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        t_backward.append(perf_counter() - t1)
    return {'forward': t_forward[2:], 'backward': t_backward[2:]}


def benchmark_sdd_torch_naive(manager, sdd, nb_vars, nb_repeats=10, device='cpu'):
    t_forward = []
    with torch.inference_mode():
        for _ in range(nb_repeats+2):
            weights, neg_weights = torch_weights(nb_vars, 'log',  device)
            t1 = perf_counter()
            eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_forward.append(perf_counter() - t1)

    t_backward = []
    for _ in range(nb_repeats + 2):
        weights, neg_weights = torch_weights(manager.var_count(), 'log',  device)
        t1 = perf_counter()
        eval_sdd_torch_naive(manager, sdd, weights, neg_weights, device).backward()
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


def torch_weights(nb_vars, semiring = 'log', device='cpu'):
    weights, neg_weights = python_weights(nb_vars, semiring)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    neg_weights = torch.tensor(neg_weights, dtype=torch.float32, device=device)
    weights.requires_grad = True
    neg_weights.requires_grad = True
    return weights, neg_weights


def python_weights(nb_vars, semiring = "log"):
    weights = [random.random() for _ in range(nb_vars)]
    neg_weights = [1-x for x in weights]
    if semiring == "log":
        weights = [math.log(x) for x in weights]
        neg_weights = [math.log(x) for x in neg_weights]
    return weights, neg_weights


def jax_weights(nb_vars, semiring = "log"):
    weights, neg_weights = python_weights(nb_vars, semiring)
    weights = jax.numpy.array(weights)
    neg_weights = jax.numpy.array(neg_weights)
    return weights, neg_weights