import math

from jax import numpy as jnp
from jax.lax import stop_gradient
from jax.ops import segment_sum, segment_max


def log1mexp(x):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = -math.log(2) < x  # x < 0
    return jnp.where(
        mask,
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def encode_input_log(pos, neg):
    if neg is None:
        neg = log1mexp(pos)

    result = jnp.stack([pos, neg], axis=1).flatten()
    constants = jnp.array([float('-inf'), 0], dtype=pos.dtype)
    return jnp.concat([constants, result])


def make_log_sum_layer(eps: float):
    def log_sum_layer(num_segments, ix_in, ix_out, x):
        x = x[ix_in]
        x_max = segment_max(stop_gradient(x), ix_out, indices_are_sorted=True, num_segments=num_segments)
        x = x - x_max[ix_out]
        x = jnp.nan_to_num(x, copy=False, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
        x = jnp.exp(x)
        x = segment_sum(x, ix_out, indices_are_sorted=True, num_segments=num_segments)
        x = jnp.log(x + eps) + x_max
        return x
    return log_sum_layer
