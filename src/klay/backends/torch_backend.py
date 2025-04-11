import math
from functools import reduce

import torch
from torch import nn

CUTOFF = -math.log(2)

def log1mexp(x, eps):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()+eps).log(),
        (-x.exp()+eps).log1p(),
    )

def negate_real(x, eps):
    return 1 - x


def encode_input(pos, neg, zero, one):
    result = torch.stack([pos, neg], dim=1).flatten()
    constants = torch.tensor([zero, one], dtype=torch.float32, device=pos.device)
    return torch.cat([constants, result])


def unroll_csr(csr):
    deltas = torch.diff(csr)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=csr.device)
    return ixs.repeat_interleave(repeats=deltas)


class KnowledgeModule(nn.Module):
    def __init__(self, pointers, csrs, semiring='real', probabilistic=False):
        super(KnowledgeModule, self).__init__()
        layers = []
        self.probabilistic = probabilistic
        sum_layer, prod_layer, self.zero, self.one, self.negate = get_semiring(semiring, probabilistic)
        for i, (ptrs, csr) in enumerate(zip(pointers, csrs)):
            ptrs = torch.as_tensor(ptrs)
            csr = torch.as_tensor(csr, dtype=torch.long)
            csr = unroll_csr(csr)
            if i % 2 == 0:
                layers.append(prod_layer(ptrs, csr))
            else:
                layers.append(sum_layer(ptrs, csr))
        self.layers = nn.Sequential(*layers)

    def forward(self, weights, neg_weights=None, eps=0):
        if neg_weights is None:
            neg_weights = self.negate(weights, eps)
        x = encode_input(weights, neg_weights, self.zero, self.one)
        return self.layers(x)

    def sparsity(self, nb_vars: int) -> float:
        sparse_params = sum(len(l.csr) for l in self.layers)
        layer_widths = [nb_vars] + [l.out_shape[0] for l in self.layers]
        dense_params  = sum(layer_widths[i] * layer_widths[i+1] for i in range(len(layer_widths) - 1))
        return sparse_params / dense_params

    def sample_pc(self):
        assert self.probabilistic
        y = torch.tensor([1])
        for layer in reversed(self.layers):
            y = layer.sample_pc(y)
        return y[2::2]


class KnowledgeLayer(nn.Module):
    def __init__(self, ptrs, csr):
        super().__init__()
        self.register_buffer('ptrs', ptrs)
        self.register_buffer('csr', csr)
        self.out_shape = (self.csr[-1].item() + 1,)
        self.in_shape = (self.ptrs.max() + 1,)

    def _scatter_forward(self, x: torch.Tensor, reduce: str):
        output = torch.empty(self.out_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.csr, src=x, reduce=reduce, include_self=False)
        return output

    def _scatter_backward(self, x: torch.Tensor, reduce: str):
        output = torch.empty(self.in_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.ptrs, src=x, reduce=reduce, include_self=False)
        return output


    def _safe_exp(self, x: torch.Tensor):
        with torch.no_grad():
            max_output = self._scatter_forward(x, "amax")
        x = x - max_output[self.csr]
        x.nan_to_num_(nan=0., posinf=float('inf'), neginf=float('-inf'))
        return torch.exp(x), max_output

    def _logsumexp_scatter(self, x: torch.Tensor, epsilon: float):
        x, max_output = self._safe_exp(x)
        output = torch.full(self.out_shape, epsilon, dtype=x.dtype, device=x.device)
        output = torch.scatter_add(output, 0, index=self.csr, src=x)
        output = torch.log(output) + max_output
        return output



class ProbabilisticKnowledgeLayer(KnowledgeLayer):
    def __init__(self, ptrs, csr):
        super().__init__(ptrs, csr)
        self.weights = nn.Parameter(torch.randn_like(ptrs, dtype=torch.float32))


class SumLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ptrs], "sum")

    def sample_pc(self, y):
        return self._scatter_backward(y[self.csr], "sum") > 0


class ProdLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ptrs], "prod")


class MinLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ptrs], "amin")


class MaxLayer(KnowledgeLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ptrs], "amax")


class LogSumLayer(KnowledgeLayer):
    def forward(self, x, epsilon=10e-16):
        return self._logsumexp_scatter(x[self.ptrs], epsilon)


class ProbabilisticSumLayer(ProbabilisticKnowledgeLayer):
    def forward(self, x):
        x = self.get_edge_weights() * x[self.ptrs]
        return self._scatter_forward(x, "sum")

    def get_edge_weights(self):
        exp_weights, _ = self._safe_exp(self.weights)
        norm = self._scatter_forward(exp_weights, "sum")
        return exp_weights / norm[self.csr]


class ProbabilisticLogSumLayer(ProbabilisticKnowledgeLayer):
    def forward(self, x, epsilon=10e-16):
        x = self.get_edge_weights(epsilon) + x[self.ptrs]
        return self._logsumexp_scatter(x, epsilon)

    def get_edge_weights(self, epsilon):
        norm = self._logsumexp_scatter(self.weights, epsilon)
        return self.weights - norm[self.csr]

    def sample_pc(self, y, epsilon=10e-16):
        weights = self.get_edge_weights(epsilon)
        gumbels = weights - (-torch.rand_like(weights).log()).log()
        samples = self._scatter_forward(gumbels, "amax")
        samples = samples[self.csr] == gumbels
        samples &= y[self.csr].to(torch.bool)
        return self._scatter_backward(samples, "sum") > 0


def get_semiring(name: str, probabilistic: bool):
    """
    For a given semiring, returns the sum and product layer,
    the zero and one elements, and a negation function.
    """
    if probabilistic:
        if name == "real":
            return ProbabilisticSumLayer, ProdLayer, 0, 1, negate_real
        if name == "log":
            return ProbabilisticLogSumLayer, SumLayer, float('-inf'), 0, log1mexp
        raise ValueError(f"Unknown probabilistic semiring {name}")
    else:
        if name == "real":
            return SumLayer, ProdLayer, 0, 1, negate_real
        elif name == "log":
            return LogSumLayer, SumLayer, float('-inf'), 0, log1mexp
        elif name == "mpe":
            return MaxLayer, ProdLayer, 0, 1, negate_real
        elif name == "godel":
            return MaxLayer, MinLayer, 0, 1, negate_real
        raise ValueError(f"Unknown semiring {name}")
