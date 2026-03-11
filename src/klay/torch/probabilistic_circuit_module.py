import torch

from .circuit_module import CircuitModule
from .probabilistic_layers import ProbabilisticCircuitLayer, ProbabilisticSumLayer, ProbabilisticLogSumLayer

_PROB_LAYER_CLASSES = {
    "sum": ProbabilisticSumLayer,
    "logsumexp": ProbabilisticLogSumLayer,
}


class ProbabilisticCircuitModule(CircuitModule):

    def __init__(self, ixs_in, ixs_out, layer_types, semiring: str = 'real'):
        if not isinstance(semiring, str):
            raise ValueError(f"ProbabilisticCircuitModule only supports named semirings {list(_PROB_LAYER_CLASSES)}, got {semiring!r}")
        super().__init__(ixs_in, ixs_out, layer_types, semiring)
        sum_reduce = self.default_semirings[semiring][0]
        self.sum_layer = _PROB_LAYER_CLASSES[sum_reduce]
        # Rebuild sum layers as probabilistic
        layers = []
        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == 0:
                layers.append(self.sum_layer(layer.ix_in, layer.ix_out))
            else:
                layers.append(layer)
        self.layers = torch.nn.Sequential(*layers)

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
        pc = ProbabilisticCircuitModule([], [], [], circuit.semiring)
        pc.layer_types = circuit.layer_types
        layers = []

        x = circuit.encode_input(x_pos, x_neg)
        for i, layer in enumerate(circuit.layers):
            if circuit.layer_types[i] == 0:
                new_layer = pc.sum_layer(layer.ix_in, layer.ix_out)
                weights = x.log() if circuit.semiring == "real" else x
                new_layer.weights.data = weights[new_layer.ix_in]
            else:
                new_layer = layer
            x = layer(x)
            layers.append(new_layer)

        pc.layers = torch.nn.Sequential(*layers)
        return pc
