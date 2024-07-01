import math

import torch
from torch_scatter import segment_csr


EPSILON = 10e-16


def log1mexp(x):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )


def encode_input(pos, neg=None):
    if neg is None:
        neg = log1mexp(pos)

    shape = (2 * pos.shape[0] + 2,) + pos.shape[1:]
    result = torch.empty(shape, dtype=torch.float32).to(pos.device)
    result[2::2] = pos
    result[3::2] = neg
    result[0] = float('-inf')
    result[1] = 0
    return result


class KnowledgeLayer(torch.nn.Module):
    def __init__(self, pointers, csrs):
        super(KnowledgeLayer, self).__init__()
        layers = []
        for i, (ptrs, csr) in enumerate(zip(pointers, csrs)):
            ptrs = torch.as_tensor(ptrs)
            csr = torch.as_tensor(csr)
            if i % 2 == 0:
                layers.append(ProductLayer(ptrs, csr))
            else:
                layers.append(SumLayer(ptrs, csr))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = encode_input(x)
        return self.layers(x)


class SumLayer(torch.nn.Module):
    def __init__(self, ptrs, csr):
        super(SumLayer, self).__init__()
        self.ptrs = ptrs
        self.csr = csr
        deltas = torch.diff(csr)
        ixs = torch.arange(len(deltas), dtype=torch.int32, device=ptrs.device)
        self.ptrs_rev = ixs.repeat_interleave(repeats=deltas)

    def forward(self, x):
        x = x[self.ptrs]
        with torch.no_grad():
            a_add = segment_csr(x, self.csr, reduce="max")
            a_sub = a_add[self.ptrs_rev]
        x = torch.exp(x - a_sub)

        x = segment_csr(x, self.csr, reduce="sum")
        x = torch.log(x + EPSILON) + a_add
        return x


class ProductLayer(torch.nn.Module):
    def __init__(self, ptrs, csr):
        super(ProductLayer, self).__init__()
        self.ptrs = ptrs
        self.csr = csr

    def forward(self, x):
        x = x[self.ptrs]
        x = segment_csr(x, self.csr, reduce='sum')
        return x
