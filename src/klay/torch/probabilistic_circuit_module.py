import torch

from .circuit_module import CircuitModule
from .layers import ProbabilisticCircuitLayer, ProbabilisticSumLayer, ProbabilisticLogSumLayer
from .layers import ProdLayer, SumLayer, negate_real, log1mexp


class ProbabilisticCircuitModule(CircuitModule):
    default_semirings = {
        "real": (ProbabilisticSumLayer, ProdLayer, 0, 1, negate_real),
        "log": (ProbabilisticLogSumLayer, SumLayer, float('-inf'), 0, log1mexp),
    }

    def sample(self):
        """ Samples from the probabilistic circuit distribution. """
        y = torch.tensor([1])
        for layer in reversed(self.layers):
            y = layer.sample(y)
        return y[2::2]

    def condition(self, x_pos, x_neg):
        x = self.encode_input(x_pos, x_neg)
        for layer in self.layers:
            x = layer.condition(x) \
                if isinstance(layer, ProbabilisticCircuitLayer) \
                else layer(x)
        return x
