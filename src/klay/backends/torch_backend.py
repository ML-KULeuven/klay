import math

import torch


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
    result = torch.empty(shape, dtype=torch.float32, device=pos.device)
    result[2::2] = pos
    result[3::2] = neg
    result[0] = float('-inf')
    result[1] = 0
    return result


def unroll_csr(csr):
    deltas = torch.diff(csr)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=csr.device)
    return ixs.repeat_interleave(repeats=deltas)


class KnowledgeLayer(torch.nn.Module):
    def __init__(self, pointers, csrs):
        super(KnowledgeLayer, self).__init__()
        layers = []
        for i, (ptrs, csr) in enumerate(zip(pointers, csrs)):
            ptrs = torch.as_tensor(ptrs)
            csr = torch.as_tensor(csr, dtype=torch.long)
            csr = unroll_csr(csr)
            if i % 2 == 0:
                layers.append(ProductLayer(ptrs, csr))
            else:
                layers.append(SumLayer(ptrs, csr))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, *x):
        x = encode_input(*x)
        return self.layers(x)


class SumLayer(torch.nn.Module):
    def __init__(self, ptrs, csr):
        super(SumLayer, self).__init__()
        self.register_buffer('ptrs', ptrs)
        self.register_buffer('csr', csr)

    def forward(self, x):
        x = x[self.ptrs]
        with torch.no_grad():
            max_output = torch.zeros(self.csr[-1]+1, dtype=x.dtype, device=x.device)
            max_output.scatter_reduce_(0, index=self.csr, src=x, reduce="amax", include_self=False)
        x = torch.exp(x - max_output[self.csr])
        x = x.nan_to_num(nan=0.0, posinf=float('inf'), neginf=float('-inf'))

        output = torch.empty(self.csr[-1]+1, dtype=x.dtype, device=x.device)
        output.fill_(EPSILON)
        output.scatter_add_(0, index=self.csr, src=x)
        output = torch.log(output) + max_output
        return output


class ProductLayer(torch.nn.Module):
    def __init__(self, ptrs, csr):
        super(ProductLayer, self).__init__()
        self.register_buffer('ptrs', ptrs)
        self.register_buffer('csr', csr)

    def forward(self, x):
        output = torch.zeros(self.csr[-1]+1, dtype=x.dtype, device=x.device)
        output.scatter_add_(0, index=self.csr, src=x[self.ptrs])
        return output
