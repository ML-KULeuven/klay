import torch
from torch import nn

from .layers import ProbabilisticCircuitLayer, get_semiring
from .utils import (
    unroll_ixs,
    set_eps,
    EPS,
    DEFAULT_EPS_VALUES_PROB,
    DEFAULT_EPS_VALUES_LOGPROB,
)


def _create_layers(sum_layer, prod_layer, ixs_in, ixs_out):
    layers = []
    for i, (ix_in, ix_out) in enumerate(zip(ixs_in, ixs_out)):
        ix_in = torch.as_tensor(ix_in, dtype=torch.long)
        ix_out = torch.as_tensor(ix_out, dtype=torch.long)
        ix_out = unroll_ixs(ix_out)
        layer = prod_layer if i % 2 == 0 else sum_layer
        layers.append(layer(ix_in, ix_out))
    return nn.Sequential(*layers)


class CircuitModule(nn.Module):
    def __init__(self, ixs_in, ixs_out, semiring='real'):
        super(CircuitModule, self).__init__()
        self.semiring = semiring
        self.sum_layer, self.prod_layer, self.zero, self.one, self.negate = \
            get_semiring(semiring, self.is_probabilistic())
        self.layers = _create_layers(self.sum_layer, self.prod_layer, ixs_in, ixs_out)

    def forward(self, x_pos, x_neg=None):
        x = self.encode_input(x_pos, x_neg)
        return self.layers(x)

    def encode_input(self, pos, neg):
        if neg is None:
            neg = self.negate(pos)
        x = torch.stack([pos, neg], dim=1).flatten()
        units = torch.tensor([self.zero, self.one], dtype=torch.float32, device=pos.device)
        return torch.cat([units, x])

    def sparsity(self, nb_vars: int) -> float:
        sparse_params = sum(len(l.ix_out) for l in self.layers)
        layer_widths = [nb_vars] + [l.out_shape[0] for l in self.layers]
        dense_params = sum(layer_widths[i] * layer_widths[i + 1] for i in range(len(layer_widths) - 1))
        return sparse_params / dense_params

    def to_pc(self, x_pos, x_neg=None):
        """ Converts the circuit into a probabilistic circuit."""
        assert self.semiring == "log" or self.semiring == "real"
        pc = ProbabilisticCircuitModule([], [], self.semiring)
        print("Making PC", pc.sum_layer, pc.sum_layer)
        layers = []

        x = self.encode_input(x_pos, x_neg)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, self.sum_layer):
                new_layer = pc.sum_layer(layer.ix_in, layer.ix_out)
                weights = x.log() if self.semiring == "real" else x
                new_layer.weights.data = weights[new_layer.ix_in]
            else:
                new_layer = layer
            x = layer(x)
            layers.append(new_layer)

        pc.layers = nn.Sequential(*layers)
        return pc

    def is_probabilistic(self) -> bool:
        """ Checks whether this circuit is probabilistic. """
        return False


class ProbabilisticCircuitModule(CircuitModule):
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

    def is_probabilistic(self) -> bool:
        """ Checks whether this circuit is probabilistic. """
        return True
