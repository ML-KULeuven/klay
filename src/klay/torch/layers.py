from functools import partial
from typing import Callable

import torch
from torch import nn

from .utils import gather_indices, scatter_logsumexp


class AbstractCircuitLayer(nn.Module):
    def __init__(self, ix_in: torch.Tensor, ix_out: torch.Tensor):
        super().__init__()
        self.register_buffer('ix_in', ix_in)
        self.register_buffer('ix_out', ix_out)
        self.out_shape = (self.ix_out[-1].item() + 1,)
        self.in_shape = (self.ix_in.max().item() + 1,)

    def sample(self, y: torch.Tensor) -> torch.Tensor:
        y = y[self.ix_out]
        output = torch.zeros(self.in_shape, dtype=y.dtype, device=y.device)
        return torch.scatter_reduce(output, 0, index=self.ix_in, src=y, reduce="amax", include_self=False)


class CircuitLayer(AbstractCircuitLayer):

    def __init__(self, ix_in: torch.Tensor, ix_out: torch.Tensor, reduce: str):
        super().__init__(ix_in, ix_out)
        self.reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = x[self.ix_in]
        out = torch.empty(self.out_shape, dtype=src.dtype, device=src.device)
        return torch.scatter_reduce(out, 0, index=self.ix_out, src=src, reduce=self.reduce, include_self=False)


class LogSumExpLayer(CircuitLayer):

    def __init__(self, ix_in: torch.Tensor, ix_out: torch.Tensor):
        super().__init__(ix_in, ix_out, reduce="amax")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scatter_logsumexp(x[self.ix_in], self.ix_out, self.out_shape)


class GatherCircuitLayer(AbstractCircuitLayer):

    def __init__(self, ix_in: torch.Tensor, ix_out: torch.Tensor, reduce_fn: Callable, fill_value: float = 0):
        super().__init__(ix_in, ix_out)
        self.reduce_fn = reduce_fn
        self.fill_value = fill_value
        order, sorted_index, positions, max_len = gather_indices(ix_out, self.out_shape[0])
        self.register_buffer('order', order)
        self.register_buffer('sorted_index', sorted_index)
        self.register_buffer('positions', positions)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sorted_src = x[self.ix_in][self.order]
        padded = x.new_full((self.out_shape[0], self.max_len), self.fill_value)
        padded[self.sorted_index, self.positions] = sorted_src
        return self.reduce_fn(padded, dim=1)


SumLayer = partial(CircuitLayer, reduce="sum")
ProdLayer = partial(CircuitLayer, reduce="prod")
MinLayer = partial(CircuitLayer, reduce="amin")
MaxLayer = partial(CircuitLayer, reduce="amax")
