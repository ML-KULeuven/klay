import numpy as np
import jax
import jax.numpy as jnp

from klay.jax.semiring import get_semiring, encode_input


def create_knowledge_layer(pointers, ix_outs, semiring, eps: float = 0):
    ixs_in = [np.array(ix_in) for ix_in in pointers]
    num_segments = [len(ix_out) - 1 for ix_out in ix_outs]  # needed for the jit
    ixs_out = [unroll_ix_out(np.array(ix_out, dtype=np.int32)) for ix_out in ix_outs]
    sum_layer, prod_layer = get_semiring(semiring, eps)
    encoder = encode_input(semiring)


    @jax.jit
    def wrapper(pos, neg=None):
        x = encoder(pos, neg)
        for i, (ix_in, ix_out) in enumerate(zip(ixs_in, ixs_out)):
            if i % 2 == 0:
                x = prod_layer(num_segments[i], ix_in, ix_out, x)
            else:
                x = sum_layer(num_segments[i], ix_in, ix_out, x)
        return x

    return wrapper


def unroll_ix_out(ix_out):
    deltas = np.diff(ix_out)
    ixs = np.arange(len(deltas), dtype=jnp.int32)
    return np.repeat(ixs, repeats=deltas)
