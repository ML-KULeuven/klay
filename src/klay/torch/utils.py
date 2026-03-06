import math

import torch

CUTOFF = -math.log(2)


def log1mexp(x, eps=1e-12):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    return torch.where(
        mask,
        (-x.clamp(min=CUTOFF).expm1() + eps).log(),
        (-x.clamp(max=CUTOFF).exp() + eps).log1p()
    )


def negate_real(x, eps):
    return 1 - x


def unroll_ixs(ixs):
    deltas = torch.diff(ixs)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=ixs.device)
    return ixs.repeat_interleave(repeats=deltas)
