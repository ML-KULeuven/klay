import math

import torch


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


def encode_input(pos, neg):
    if neg is None:
        neg = log1mexp(pos)

    result = torch.stack([pos, neg], dim=1).flatten()
    constants = torch.tensor([float('-inf'), 0], dtype=torch.float32, device=pos.device)
    return torch.cat([constants, result])


def unroll_csr(csr):
    deltas = torch.diff(csr)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=csr.device)
    return ixs.repeat_interleave(repeats=deltas)


class KnowledgeLayer(torch.nn.Module):
    def __init__(self, pointers, csrs, semiring='real'):
        super(KnowledgeLayer, self).__init__()
        layers = []
        sum_layer, prod_layer = get_semiring(semiring)
        for i, (ptrs, csr) in enumerate(zip(pointers, csrs)):
            ptrs = torch.as_tensor(ptrs)
            csr = torch.as_tensor(csr, dtype=torch.long)
            csr = unroll_csr(csr)
            if i % 2 == 0:
                layers.append(prod_layer(ptrs, csr))
            else:
                layers.append(sum_layer(ptrs, csr))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, weights, neg_weights=None):
        x = encode_input(weights, neg_weights)
        return self.layers(x)


class SumLayer(torch.nn.Module):
    def __init__(self, ptrs, csr):
        super().__init__()
        self.register_buffer('ptrs', ptrs)
        self.register_buffer('csr', csr)
        self.out_shape = (self.csr[-1].item() + 1,)

    def forward(self, x):
        output = torch.zeros(self.out_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_add(output, 0, index=self.csr, src=x[self.ptrs])
        return output


class ProdLayer(torch.nn.Module):
    def __init__(self, ptrs, csr):
        super().__init__()
        self.register_buffer('ptrs', ptrs)
        self.register_buffer('csr', csr)
        self.out_shape = (self.csr[-1].item() + 1,)

    def forward(self, x):
        output = torch.zeros(self.out_shape, dtype=x.dtype, device=x.device)
        output = torch.scatter_reduce(output, 0, index=self.csr, src=x[self.ptrs], reduce="prod", include_self=False)
        return output


class LogSumLayer(torch.nn.Module):
    def __init__(self, ptrs, csr):
        super().__init__()
        self.register_buffer('ptrs', ptrs)
        self.register_buffer('csr', csr)
        self.out_shape = (self.csr[-1].item() + 1,)

    def forward(self, x, epsilon=10e-16):
        x = x[self.ptrs]
        with torch.no_grad():
            max_output = torch.empty(self.out_shape, dtype=x.dtype, device=x.device)
            max_output = torch.scatter_reduce(max_output, 0, index=self.csr, src=x, reduce="amax", include_self=False)
        x = x - max_output[self.csr]
        x.nan_to_num_(nan=0., posinf=float('inf'), neginf=float('-inf'))
        x = torch.exp(x)

        output = torch.full(self.out_shape, epsilon, dtype=x.dtype, device=x.device)
        output = torch.scatter_add(output, 0, index=self.csr, src=x)
        output = torch.log(output) + max_output
        return output


def get_semiring(name: str):
    if name == "real":
        return SumLayer, ProdLayer
    elif name == "log":
        return LogSumLayer, SumLayer
    else:
        raise ValueError(f"Unknown semiring {name}")
