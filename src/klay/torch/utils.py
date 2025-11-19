import math

import torch

CUTOFF = -math.log(2)


def log1mexp(x, eps=10e-12):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    out = torch.empty_like(x)
    out[mask] = (-x[mask].expm1() + eps).log()
    out[~mask] = (-x[~mask].exp() + eps).log1p()
    return out


def negate_real(x, eps):
    return 1 - x


def unroll_ixs(ixs):
    deltas = torch.diff(ixs)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=ixs.device)
    return ixs.repeat_interleave(repeats=deltas)
