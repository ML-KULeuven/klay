import torch

from .circuit_module import CircuitModule
from .layers import ProdLayer, SumLayer
from .probabilistic_layers import ProbabilisticCircuitLayer, ProbabilisticSumLayer, ProbabilisticLogSumLayer
from .utils import negate_real, log1mexp

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

    @staticmethod
    def from_circuit(circuit: CircuitModule, x_pos, x_neg=None):
        """ Converts the circuit into a probabilistic circuit."""
        assert circuit.semiring == "log" or circuit.semiring == "real"
        pc = ProbabilisticCircuitModule([], [], circuit.semiring)
        layers = []

        x = circuit.encode_input(x_pos, x_neg)
        for i, layer in enumerate(circuit.layers):
            if isinstance(layer, circuit.sum_layer):
                new_layer = pc.sum_layer(layer.ix_in, layer.ix_out, layer._eps)
                weights = x.log() if circuit.semiring == "real" else x
                new_layer.weights.data = weights[new_layer.ix_in]
            else:
                new_layer = layer
            x = layer(x)
            layers.append(new_layer)

        pc.layers = torch.nn.Sequential(*layers)
        return pc