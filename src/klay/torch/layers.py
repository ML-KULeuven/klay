from abc import ABC
from functools import partial

import torch
from torch import nn


class AbstractCircuitLayer(nn.Module, ABC):
    def __init__(self, ix_in: torch.Tensor, ix_out: torch.Tensor):
        super().__init__()
        self.register_buffer('ix_in', ix_in)
        self.register_buffer('ix_out', ix_out)
        self.out_shape = (self.ix_out[-1].item() + 1,)
        self.in_shape = (self.ix_in.max().item() + 1,)

    def sample(self, y: torch.Tensor):
        y = y[self.ix_out]
        output = torch.zeros(self.in_shape, dtype=y.dtype, device=y.device)
        output = torch.scatter_reduce(output, 0, index=self.ix_in, src=y, reduce="amax", include_self=False)
        return output


class ScatterCircuitLayer(AbstractCircuitLayer):

    def __init__(self, ix_in: torch.Tensor, ix_out: torch.Tensor, reduce: str):
        super().__init__(ix_in, ix_out)
        self._reduce = reduce

    def forward(self, x: torch.Tensor):
        x = x[self.ix_in]
        output = torch.empty(self.out_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.ix_out, src=x, reduce=self._reduce, include_self=False)
        return output




SumLayer = partial(ScatterCircuitLayer, reduce="sum")
ProdLayer = partial(ScatterCircuitLayer, reduce="prod")
MinLayer = partial(ScatterCircuitLayer, reduce="amin")
MaxLayer = partial(ScatterCircuitLayer, reduce="amax")


class LogSumScatterLayer(ScatterCircuitLayer):

    def _safe_exp(self, x: torch.Tensor):
        max_output = self._scatter_forward(x.detach(), "amax")
        x = x - max_output[self.ix_out]
        x.nan_to_num_(nan=0., posinf=float('inf'), neginf=float('-inf'))
        return torch.exp(x), max_output

    def forward(self, x: torch.Tensor):
        x = x[self.ix_in]
        x, max_output = self._safe_exp(x)
        output = torch.full(self.out_shape, torch.finfo(x.dtype).eps, dtype=x.dtype, device=x.device)
        output = torch.scatter_add(output, 0, index=self.ix_out, src=x)
        output = torch.log(output) + max_output
        return output
