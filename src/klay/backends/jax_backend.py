import math
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_max, segment_sum
from jax.lax import stop_gradient


EPSILON = 10e-16


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


def encode_input(pos, neg=None):
    if neg is None:
        neg = log1mexp(pos)

    shape = (2 * pos.shape[0] + 2,) + pos.shape[1:]
    result = jnp.empty(shape, dtype=jnp.float32)
    result = result.at[2::2].set(pos)
    result = result.at[3::2].set(neg)
    result = result.at[0].set(float('-inf'))
    result = result.at[1].set(0)
    return result


def create_knowledge_layer(pointers, csrs):
    pointers = [np.array(ptrs) for ptrs in pointers]
    num_segments = [len(csr) - 1 for csr in csrs]  # needed for the jit
    csrs = [unroll_csr(np.array(csr, dtype=np.int32)) for csr in csrs]

    @jax.jit
    def wrapper(x):
        x = encode_input(x)
        for i, (ptrs, csr) in enumerate(zip(pointers, csrs)):
            if i % 2 == 0:
                x = product_layer(num_segments[i], ptrs, csr, x)
            else:
                x = sum_layer(num_segments[i], ptrs, csr, x)
        return x

    return wrapper


def unroll_csr(csr):
    deltas = np.diff(csr)
    ixs = np.arange(len(deltas), dtype=jnp.int32)
    return np.repeat(ixs, repeats=deltas)


@partial(jax.jit, static_argnums=(0,), inline=True)
def sum_layer(num_segments, ptrs, csr, x):
    x = x[ptrs]
    a_add = segment_max(stop_gradient(x), csr, indices_are_sorted=True, num_segments=num_segments)
    x = jnp.exp(x - a_add[csr])
    x = jnp.nan_to_num(x, copy=False, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
    # x = x.at[jnp.isnan(x)].set(0)
    x = segment_sum(x, csr, indices_are_sorted=True, num_segments=num_segments)
    x = jnp.log(x + EPSILON) + a_add
    return x


@partial(jax.jit, static_argnums=(0,), inline=True)
def product_layer(num_segments, ptrs, csr, x):
    x = x[ptrs]
    x = segment_sum(x, csr, num_segments=num_segments)
    return x
