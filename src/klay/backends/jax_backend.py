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
    constants = jnp.array([float("-inf"), 0], dtype=jnp.float32)
    return jnp.concat([constants, result])


def encode_input_real(pos, neg):
    if neg is None:
        neg = 1 - pos

    result = jnp.stack([pos, neg], axis=1).flatten()
    constants = jnp.array(
        [
            0.0,
            1,
        ],
        dtype=jnp.float32,
    )
    return jnp.concat([constants, result])


def create_knowledge_layer(pointers, csrs, semiring):
    pointers = [np.array(ptrs) for ptrs in pointers]
    num_segments = [len(csr) - 1 for csr in csrs]  # needed for the jit
    csrs = [unroll_csr(np.array(csr, dtype=np.int32)) for csr in csrs]
    sum_layer, prod_layer = get_semiring(semiring)
    encode_input = {"log": encode_input_log, "real": encode_input_real}[semiring]

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


def exp_max(num_segments, csr, x):
    x_max = segment_max(
        stop_gradient(x), csr, indices_are_sorted=True, num_segments=num_segments
    )
    x = x - x_max[csr]
    x = jnp.nan_to_num(
        x, copy=False, nan=0.0, posinf=float("inf"), neginf=float("-inf")
    )
    x = jnp.exp(x)
    return x, x_max


@jax.custom_jvp
def log_sum_layer(num_segments, ptrs, csr, x):
    x = x[ptrs]
    x, x_max = exp_max(num_segments, csr, x)
    x = segment_sum(x, csr, indices_are_sorted=True, num_segments=num_segments)
    x = jnp.log(x + EPSILON) + x_max
    return x


@log_sum_layer.defjvp
def log_prod_layer_jvp(num_segments, ptrs, csr, p_in, d_in):
    p_out = log_sum_layer(num_segments, ptrs, csr, p_in)

    sign_d_in, mag_d_in = d_in
    sign_d = 1 - 2 * sign_d_in
    mag_d = mag_d_in - p_in + p_out[csr]

    mag_d, mag_d_max = exp_max(num_segments, csr, mag_d)
    mag_d = mag_d * sign_d
    mag_d = segment_sum(mag_d, csr, indices_are_sorted=True, num_segments=num_segments)

    sign_d_out = (jnp.sign(-mag_d) + 1) // 2
    mag_d_out = jnp.log(jnp.abs(mag_d) + EPSILON) + mag_d_max

    return p_out, (sign_d_out, mag_d_out)


@jax.custom_jvp
def log_prod_layer(num_segments, ptrs, csr, x):
    return sum_layer(num_segments, ptrs, csr, x)


@log_prod_layer.defjvp
def log_prod_layer_jvp(num_segments, ptrs, csr, p_in, d_in):
    p_out = log_prod_layer(num_segments, ptrs, csr, p_in)

    sign_d_in, mag_d_in = d_in
    d_out_sign = sum_layer(num_segments, ptrs, csr, sign_d_in) % 2
    d_out_mag = sum_layer(num_segments, ptrs, csr, mag_d_in)

    return p_out, (d_out_sign, d_out_mag)


def sum_layer(num_segments, ptrs, csr, x):
    return segment_sum(x[ptrs], csr, num_segments=num_segments, indices_are_sorted=True)


def prod_layer(num_segments, ptrs, csr, x):
    return segment_prod(
        x[ptrs], csr, num_segments=num_segments, indices_are_sorted=True
    )


def get_semiring(name: str):
    if name == "real":
        return sum_layer, prod_layer
    elif name == "log":
        return log_sum_layer, log_prod_layer
    else:
        raise ValueError(f"Unknown semiring {name}")
