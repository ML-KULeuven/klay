import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_max, segment_sum, segment_prod
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


def encode_input_log(pos, neg):
    if neg is None:
        neg = log1mexp(pos)

    result = jnp.stack([pos, neg], axis=1).flatten()
    constants = jnp.array([float('-inf'), 0], dtype=jnp.float32)
    return jnp.concat([constants, result])


def encode_input_real(pos, neg):
    if neg is None:
        neg = 1 - pos

    result = jnp.stack([pos, neg], axis=1).flatten()
    constants = jnp.array([0., 1,], dtype=jnp.float32)
    return jnp.concat([constants, result])



def create_knowledge_layer(pointers, csrs, semiring):
    pointers = [np.array(ptrs) for ptrs in pointers]
    num_segments = [len(csr) - 1 for csr in csrs]  # needed for the jit
    csrs = [unroll_csr(np.array(csr, dtype=np.int32)) for csr in csrs]
    sum_layer, prod_layer = get_semiring(semiring)
    encode_input = {'log': encode_input_log, 'real': encode_input_real}[semiring]

    @jax.jit
    def wrapper(pos, neg=None):
        x = encode_input(pos, neg)
        for i, (ptrs, csr) in enumerate(zip(pointers, csrs)):
            if i % 2 == 0:
                x = prod_layer(num_segments[i], ptrs, csr, x)
            else:
                x = sum_layer(num_segments[i], ptrs, csr, x)
        return x

    return wrapper


def unroll_csr(csr):
    deltas = np.diff(csr)
    ixs = np.arange(len(deltas), dtype=jnp.int32)
    return np.repeat(ixs, repeats=deltas)


def log_sum_layer(num_segments, ptrs, csr, x):
    x = x[ptrs]
    x_max = segment_max(stop_gradient(x), csr, indices_are_sorted=True, num_segments=num_segments)
    x = x - x_max[csr]
    x = jnp.nan_to_num(x, copy=False, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
    x = jnp.exp(x)
    x = segment_sum(x, csr, indices_are_sorted=True, num_segments=num_segments)
    x = jnp.log(x + EPSILON) + x_max
    return x


def sum_layer(num_segments, ptrs, csr, x):
    return segment_sum(x[ptrs], csr, num_segments=num_segments, indices_are_sorted=True)


def prod_layer(num_segments, ptrs, csr, x):
    return segment_prod(x[ptrs], csr, num_segments=num_segments, indices_are_sorted=True)


def get_semiring(name: str):
    if name == 'real':
        return sum_layer, prod_layer
    elif name == 'log':
        return log_sum_layer, sum_layer
    else:
        raise ValueError(f"Unknown semiring {name}")