import torch
from torch import nn

from .layers import AbstractCircuitLayer
from .utils import scatter_logsumexp


class ProbabilisticCircuitLayer(AbstractCircuitLayer):
    def __init__(self, ix_in, ix_out):
        super().__init__(ix_in, ix_out)
        self.weights = nn.Parameter(torch.randn_like(ix_in, dtype=torch.float32))

    def get_edge_weights(self):
        return self.get_log_edge_weights().exp()

    def get_log_edge_weights(self):
        norm = scatter_logsumexp(self.weights, self.ix_out, self.out_shape)
        return self.weights - norm[self.ix_out]

    def renorm_weights(self, x):
        with torch.no_grad():
            self.weights.data = self.get_log_edge_weights() + x

    def sample(self, y):
        weights = self.get_log_edge_weights()
        noise = -(-torch.log(torch.rand_like(weights))).log()
        gumbels = weights + noise
        max_vals = torch.full(self.out_shape, float('-inf'), dtype=gumbels.dtype, device=gumbels.device)
        max_vals = torch.scatter_reduce(max_vals, 0, index=self.ix_out, src=gumbels.detach(), reduce="amax", include_self=False)
        samples = max_vals[self.ix_out] == gumbels
        samples &= y[self.ix_out].to(torch.bool)
        result = torch.zeros(self.in_shape, dtype=torch.long, device=y.device)
        result = torch.scatter_reduce(result, 0, index=self.ix_in, src=samples.long(), reduce="sum", include_self=False)
        return result > 0


class ProbabilisticSumLayer(ProbabilisticCircuitLayer):
    def forward(self, x):
        weighted = self.get_edge_weights() * x[self.ix_in]
        out = torch.zeros(self.out_shape, dtype=weighted.dtype, device=x.device)
        return torch.scatter_add(out, 0, index=self.ix_out, src=weighted)

    def condition(self, x):
        y = self.forward(x)
        self.renorm_weights(x[self.ix_in].log())
        return y


class ProbabilisticLogSumLayer(ProbabilisticCircuitLayer):
    def forward(self, x):
        return scatter_logsumexp(self.get_log_edge_weights() + x[self.ix_in], self.ix_out, self.out_shape)

    def condition(self, x):
        y = self.forward(x)
        self.renorm_weights(x[self.ix_in])
        return y
