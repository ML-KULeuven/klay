import torch
from torch import nn

from .layers import get_semiring, ProbabilisticCircuitLayer
from .utils import unroll_ixs


class CircuitModule(nn.Module):
    def __init__(self, ixs_in, ixs_out, semiring: str = 'real', eps: float = 0):
        super(CircuitModule, self).__init__()
        self.semiring = semiring
        self._eps = 0

        self.sum_layer, self.prod_layer, self.zero, self.one, self.negate = \
            get_semiring(semiring, self.is_probabilistic())

        layers = []
        for i, (ix_in, offsets) in enumerate(zip(ixs_in, ixs_out)):
            ix_in = torch.as_tensor(ix_in, dtype=torch.long)
            offsets = torch.as_tensor(offsets, dtype=torch.long)
            ix_out = unroll_ixs(offsets)
            layer = self.prod_layer if i % 2 == 0 else self.sum_layer
            layers.append(layer(ix_in, offsets, ix_out, eps))
        self.layers = nn.Sequential(*layers)


    def forward(self, x_pos, x_neg=None):
        x = self.encode_input(x_pos, x_neg)
        x = self.layers(x)
        return x.mT if x.dim() == 2 else x

    def encode_input(self, pos, neg):
        if neg is None:
            neg = self.negate(pos, self._eps)
        units = torch.tensor([self.zero, self.one], dtype=pos.dtype, device=pos.device)
        x = torch.stack([pos, neg], dim=-1).flatten(-2)
        units = units.expand(*x.shape[:-1], 2)
        x = torch.cat([units, x], dim=-1)
        return x.mT if x.dim() == 2 else x

    def sparsity(self, nb_vars: int) -> float:
        sparse_params = sum(len(layer.ix_out) for layer in self.layers)
        layer_widths = [nb_vars] + [layer.out_shape[0] for layer in self.layers]
        dense_params = sum(layer_widths[i] * layer_widths[i + 1] for i in range(len(layer_widths) - 1))
        return sparse_params / dense_params

    def to_pc(self, x_pos, x_neg=None):
        """ Converts the circuit into a probabilistic circuit."""
        assert self.semiring == "log" or self.semiring == "real"
        pc = ProbabilisticCircuitModule([], [], self.semiring)
        layers = []

        x = self.encode_input(x_pos, x_neg)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, self.sum_layer):
                new_layer = pc.sum_layer(layer.ix_in, layer.offsets, layer.ix_out, layer._eps)
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
