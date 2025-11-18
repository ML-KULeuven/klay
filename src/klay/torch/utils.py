import math

import torch


# Default epsilon values for different precisions
DEFAULT_EPS_VALUES_PROB = {
    torch.float16: 1e-4,
    torch.bfloat16: 1e-3,
    torch.float32: 1e-8,
    torch.float64: 1e-15,
}

DEFAULT_EPS_VALUES_LOGPROB = {
    torch.float16: 1e-4,
    torch.bfloat16: 1e-3,
    torch.float32: 1e-16,
    torch.float64: 1e-30,
}

# Global epsilon constant - used for all numerical stability operations
EPS = 1e-16

CUTOFF = -math.log(2)


def set_eps(eps: float):
    """Set global epsilon value for numerical stability in operations."""
    global EPS
    EPS = eps


def log1mexp(x):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1() + EPS).log(),
        (-x.exp() + EPS).log1p(),
    )


def negate_real(x):
    return 1 - x


def unroll_ixs(ixs):
    deltas = torch.diff(ixs)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=ixs.device)
    return ixs.repeat_interleave(repeats=deltas)
