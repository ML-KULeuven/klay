"""SpringMV implementation of Klay's JAX circuit reductions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from klay.jax.semiring import encode_input


def _load_springmv():
    try:
        import springmv
    except ModuleNotFoundError as error:
        if error.name == "springmv":
            raise ImportError(
                "The SpringMV backend requires the optional 'springmv' "
                "package. Install it from "
                "https://github.com/pedrozudo/springmv. Building SpringMV "
                "requires the CUDA toolkit and a C++ compiler."
            ) from None
        raise
    return springmv


def _int32_array(value: Any, name: str) -> jax.Array:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        int32 = np.iinfo(np.int32)
        if minimum < int32.min or maximum > int32.max:
            raise ValueError(f"{name} entries must fit in int32")
    return jnp.asarray(array, dtype=jnp.int32)


def _first_layer_width(column_indices: np.ndarray) -> int:
    """Recover ``[False, True, pos_1, neg_1, ...]`` input width."""
    if column_indices.size == 0:
        return 2
    maximum = int(column_indices.max())
    if maximum < 2:
        return 2
    return 2 * (maximum // 2 + 1)


def _make_structures(
    springmv,
    pointers: Sequence[Any],
    row_offsets: Sequence[Any],
):
    if len(pointers) != len(row_offsets):
        raise ValueError("pointers and row_offsets must have equal lengths")
    if not pointers:
        return ()

    first_columns = np.asarray(pointers[0])
    num_cols = _first_layer_width(first_columns)
    structures = []
    for layer, (columns, offsets) in enumerate(zip(pointers, row_offsets)):
        structure = springmv.CsrStructure.from_arrays(
            _int32_array(offsets, f"row_offsets[{layer}]"),
            _int32_array(columns, f"pointers[{layer}]"),
            num_cols=num_cols,
        )
        structures.append(structure)
        num_cols = structure.num_rows
    return tuple(structures)


def create_springmv_knowledge_layer(
    pointers,
    row_offsets,
    *,
    semiring: str,
    eps: float = 0,
):
    """Construct a jitted Klay evaluator backed by SpringMV CUDA kernels."""
    if semiring in ("real", "mpe"):
        raise NotImplementedError(
            f"The SpringMV backend does not support the {semiring!r} "
            "semiring. Its product reduction is not implemented for the "
            "'real' and 'mpe' semirings."
        )
    if semiring not in ("log", "godel"):
        raise ValueError(f"Unknown semiring {semiring}")
    if semiring == "log" and eps != 0:
        raise NotImplementedError(
            "The SpringMV log semiring does not support a nonzero eps"
        )

    springmv = _load_springmv()
    structures = _make_structures(springmv, pointers, row_offsets)
    encoder = encode_input(semiring)

    if semiring == "log":
        sum_reduction = springmv.csr_logsumexp
        product_reduction = springmv.csr_spmv
    else:
        sum_reduction = springmv.csr_maxplus
        product_reduction = springmv.csr_minplus

    @jax.jit
    def wrapper(pos, neg=None):
        x = encoder(pos, neg)
        if structures:
            # Native JAX gathers only referenced literals and therefore
            # accepts trailing variables that are absent from the circuit.
            # SpringMV requires an exact matrix input width, so discard the
            # same unused tail explicitly.
            x = x[: structures[0].num_cols]
        for layer, structure in enumerate(structures):
            reduction = (
                product_reduction if layer % 2 == 0 else sum_reduction
            )
            x = reduction(structure, x)
        return x

    return wrapper
