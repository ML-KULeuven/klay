import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from klay.jax.semiring import get_semiring, encode_input
from klay.jax.semiring.real import sum_layer as _plain_sum_layer


def create_knowledge_layer(pointers, ix_outs, semiring):
    ixs_in = [np.array(ix_in) for ix_in in pointers]
    num_segments = [len(ix_out) - 1 for ix_out in ix_outs]  # needed for the jit
    ixs_out = [unroll_ix_out(np.array(ix_out, dtype=np.int32)) for ix_out in ix_outs]
    sum_layer, prod_layer = get_semiring(semiring)
    encoder = encode_input(semiring)

    # Pre-build BCOO sparse matrices for layers using plain sum_layer (segment_sum).
    # The loop is unrolled at JIT trace time, so Python-level dispatch is fine.
    layers = []
    for i, (ix_in, ix_out) in enumerate(zip(ixs_in, ixs_out)):
        fn = prod_layer if i % 2 == 0 else sum_layer
        if fn is _plain_sum_layer:
            n_out = num_segments[i]
            n_in = int(ix_in.max()) + 1
            indices = np.stack([ix_out, ix_in], axis=1)
            values = np.ones(len(ix_in), dtype=np.float32)
            A = jsparse.BCOO((values, indices), shape=(n_out, n_in))
            layers.append(lambda x, A=A, n_in=n_in: A @ x[:n_in])
        else:
            ns, ii, io = num_segments[i], ix_in, ix_out
            layers.append(lambda x, fn=fn, ns=ns, ii=ii, io=io: fn(ns, ii, io, x))

    @jax.jit
    def wrapper(pos, neg=None):
        x = encoder(pos, neg)
        for layer in layers:
            x = layer(x)
        return x

    return wrapper


def unroll_ix_out(ix_out):
    deltas = np.diff(ix_out)
    ixs = np.arange(len(deltas), dtype=jnp.int32)
    return np.repeat(ixs, repeats=deltas)
