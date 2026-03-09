import torch
from torch import nn

from .layers import SumLayer, ProdLayer, LogSumScatterLayer, MaxLayer, MinLayer
from .utils import unroll_ixs, negate_real, log1mexp


class CircuitModule(nn.Module):
    default_semirings = {
        "real": (SumLayer, ProdLayer, 0, 1, negate_real),
        "log": (LogSumScatterLayer, SumLayer, float('-inf'), 0, log1mexp),
        "mpe": (MaxLayer, ProdLayer, 0, 1, negate_real),
        "godel": (MaxLayer, MinLayer, 0, 1, negate_real),
    }

    def __init__(self, ixs_in, ixs_out, semiring: str | tuple = 'real'):
        super().__init__()
        self.semiring = semiring
        if isinstance(semiring, str):
            self.sum_layer, self.prod_layer, self.zero, self.one, self.negate = self.default_semirings[semiring]
        else:
            self.sum_layer, self.prod_layer, self.zero, self.one, self.negate = semiring

        layers = []
        for i, (ix_in, ix_out) in enumerate(zip(ixs_in, ixs_out)):
            ix_in = torch.as_tensor(ix_in, dtype=torch.long)
            ix_out = torch.as_tensor(ix_out, dtype=torch.long)
            ix_out = unroll_ixs(ix_out)
            layer = self.prod_layer if i % 2 == 0 else self.sum_layer
            layers.append(layer(ix_in, ix_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x_pos, x_neg=None):
        x = self.encode_input(x_pos, x_neg)
        return self.layers(x)

    def encode_input(self, pos, neg):
        if neg is None:
            neg = self.negate(pos)
        x = torch.stack([pos, neg], dim=1).flatten()
        units = torch.tensor([self.zero, self.one], dtype=pos.dtype, device=pos.device)
        return torch.cat([units, x])

    def sparsity(self, nb_vars: int) -> float:
        sparse_params = sum(len(layer.ix_out) for layer in self.layers)
        layer_widths = [nb_vars] + [layer.out_shape[0] for layer in self.layers]
        dense_params = sum(layer_widths[i] * layer_widths[i + 1] for i in range(len(layer_widths) - 1))
        return sparse_params / dense_params

