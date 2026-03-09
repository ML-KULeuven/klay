import torch
from torch import nn

from .layers import AbstractCircuitLayer

class ProbabilisticCircuitLayer(AbstractCircuitLayer):
    def __init__(self, ix_in, ix_out, eps):
        super().__init__(ix_in, ix_out, eps)
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
        samples = self._scatter_forward(gumbels, "amax")
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
