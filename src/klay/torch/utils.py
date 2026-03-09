import math

import torch

CUTOFF = -math.log(2)


def log1mexp(x):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    eps = torch.finfo(x.dtype).eps
    return torch.where(
        mask,
        (-x.clamp(min=CUTOFF).expm1() + eps).log(),
        (-x.clamp(max=CUTOFF).exp() + eps).log1p()
    )


def negate_real(x):
    return 1 - x


def gather_indices(index: torch.Tensor, num_groups: int):
    """Compute indices for gathering sparse values into a dense 2D (num_groups, max_group_size) tensor."""
    counts = torch.bincount(index, minlength=num_groups)
    max_len = counts.max().item()
    order = index.argsort(stable=True)
    sorted_index = index[order]
    offsets = counts.cumsum(0).roll(1)
    offsets[0] = 0
    positions = torch.arange(len(index), dtype=torch.long) - offsets[sorted_index]
    return order, sorted_index, positions, max_len


def scatter_logsumexp(src: torch.Tensor, index: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    max_vals = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)
    max_vals = torch.scatter_reduce(max_vals, 0, index=index, src=src.detach(), reduce="amax", include_self=True)
    exp_sum = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    exp_sum = torch.scatter_add(exp_sum, 0, index=index, src=(src - max_vals[index]).exp())
    return torch.log(exp_sum) + max_vals


def unroll_ixs(ixs):
    deltas = torch.diff(ixs)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=ixs.device)
    return ixs.repeat_interleave(repeats=deltas)
