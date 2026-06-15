import torch
from torch import nn

from .utils import negate_real, log1mexp


class CircuitLayer(nn.Module):
    def __init__(self, ix_in, offsets, ix_out, eps):
        super().__init__()
        self.register_buffer('ix_in', ix_in)        # per-edge child index
        self.register_buffer('offsets', offsets)    # CSR segment boundaries, len = out_width + 1
        self.register_buffer('ix_out', ix_out)      # per-edge segment id (offsets, unrolled)
        self.out_shape = (self.offsets.shape[0] - 1,)
        self.in_shape = (self.ix_in.max().item() + 1,)
        self._eps = eps

    def _scatter_forward(self, x: torch.Tensor, reduce: str):
        if reduce == "logsumexp":
            return self._scatter_logsumexp_forward(x)
        if x.is_floating_point():
            return torch.segment_reduce(x, reduce, offsets=self.offsets, axis=0, unsafe=True)
        # segment_reduce has no integer kernel; fall back to scatter for integer evidence.
        output = torch.empty(self.out_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.ix_out, src=x, reduce=reduce, include_self=False)
        return output

    def _scatter_backward(self, x: torch.Tensor, reduce: str):
        output = torch.zeros(self.in_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.ix_in, src=x, reduce=reduce, include_self=False)
        return output

    def _safe_exp(self, x: torch.Tensor):
        max_output = self._scatter_forward(x.detach(), "max")
        x = x - max_output[self.ix_out]
        x.nan_to_num_(nan=0., posinf=float('inf'), neginf=float('-inf'))
        return torch.exp(x), max_output

    def _scatter_logsumexp_forward(self, x: torch.Tensor):
        x, max_output = self._safe_exp(x)
        output = torch.segment_reduce(x, "sum", offsets=self.offsets, axis=0, unsafe=True)
        output = output + self._eps
        output = output.log() + max_output
        return output

    def sample(self, y):
        return self._scatter_backward(y[self.ix_out], "max")


class SumLayer(CircuitLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "sum")


class ProdLayer(CircuitLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "prod")


class MinLayer(CircuitLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "min")


class MaxLayer(CircuitLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "max")


class LogSumLayer(CircuitLayer):
    def forward(self, x):
        return self._scatter_forward(x[self.ix_in], "logsumexp")


class ProbabilisticCircuitLayer(CircuitLayer):
    def __init__(self, ix_in, offsets, ix_out, eps):
        super().__init__(ix_in, offsets, ix_out, eps)
        self.weights = nn.Parameter(torch.randn_like(ix_in, dtype=torch.float32))

    def get_edge_weights(self):
        exp_weights, _ = self._safe_exp(self.weights)
        norm = self._scatter_forward(exp_weights, "sum")
        return exp_weights / norm[self.ix_out]

    def renorm_weights(self, x):
        with torch.no_grad():
            self.weights.data = self.get_log_edge_weights() + x

    def get_log_edge_weights(self):
        norm = self._scatter_logsumexp_forward(self.weights)
        return self.weights - norm[self.ix_out]

    def sample(self, y):
        weights = self.get_log_edge_weights()
        noise = -(-torch.log(torch.rand_like(weights) + self._eps) + self._eps).log()
        gumbels = weights + noise
        samples = self._scatter_forward(gumbels, "max")
        samples = samples[self.ix_out] == gumbels
        samples &= y[self.ix_out].to(torch.bool)
        return self._scatter_backward(samples, "sum") > 0


class ProbabilisticSumLayer(ProbabilisticCircuitLayer):
    def forward(self, x):
        x = self.get_edge_weights() * x[self.ix_in]
        return self._scatter_forward(x, "sum")

    def condition(self, x):
        x2 = self.forward(x)
        self.renorm_weights(x[self.ix_in].log())
        return x2


class ProbabilisticLogSumLayer(ProbabilisticCircuitLayer):
    def forward(self, x):
        x = self.get_log_edge_weights() + x[self.ix_in]
        return self._scatter_logsumexp_forward(x)

    def condition(self, x):
        y = self.forward(x)
        self.renorm_weights(x[self.ix_in])
        return y


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
