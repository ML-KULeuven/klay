from jax import numpy as jnp
from jax.ops import segment_sum, segment_prod


def encode_input_real(pos, neg):
    if neg is None:
        neg = 1 - pos

    result = jnp.stack([pos, neg], axis=1).flatten()
    constants = jnp.array([0., 1,], dtype=pos.dtype)
    return jnp.concat([constants, result])


def sum_layer(num_segments, ix_in, ix_out, x):
    return segment_sum(x[ix_in], ix_out, num_segments=num_segments, indices_are_sorted=True)


def prod_layer(num_segments, ix_in, ix_out, x):
    return segment_prod(x[ix_in], ix_out, num_segments=num_segments, indices_are_sorted=True)
