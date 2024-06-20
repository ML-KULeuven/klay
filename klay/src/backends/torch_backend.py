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


def encode_input(pos, neg=None):
    if neg is None:
        neg = log1mexp(pos)

    shape = (2 * pos.shape[0] + 2,) + pos.shape[1:]
    result = torch.empty(shape)
    result[2::2] = pos
    result[3::2] = neg
    result[0] = float('-inf')
    result[1] = 0
    return result


class KnowledgeLayer(torch.nn.Module):
    def __init__(self, layers):
        super(KnowledgeLayer, self).__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            if i % 2 == 0:
                self.layers.append(ProductLayer(layer))
            else:
                self.layers.append(SumLayer(layer))
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = encode_input(x)
        return self.layers(x)


class SumLayer(torch.nn.Module):
    def __init__(self, indices):
        super(SumLayer, self).__init__()
        self.indices = indices

    def forward(self, x):
        result = torch.logsumexp(x[self.indices], dim=1)
        return result


class ProductLayer(torch.nn.Module):
    def __init__(self, indices):
        super(ProductLayer, self).__init__()
        self.indices = indices

    def forward(self, x):
        result = x[self.indices].sum(dim=1)
        return result
