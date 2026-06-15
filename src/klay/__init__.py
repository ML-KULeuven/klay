# noinspection PyUnresolvedReferences
from .klay_ext import (
        Circuit,
        NodePtr,
        check_sdnnf,
        check_decomposability,
        check_smooth,
        SDNNFResult,
        SDNNFViolation,
)
NodePtr.__module__ = "klay"

from collections.abc import Sequence
import tempfile
import os
from pathlib import Path


def to_torch_module(self: Circuit, semiring: str = "log", probabilistic: bool = False, eps: float = 0, compile=True):
    """
    Convert the circuit into a PyTorch module.

    :param semiring:
        The semiring in which the circuit should be evaluated. Supported options are :code:`"log"`, :code:`"real"`, :code:`"mpe"`, or :code:`"godel"`.
    :param probabilistic:
        If enabled, construct a probabilistic circuit instead of an arithmetic circuit.
        This means the inputs to a sum node are multiplied by a probability, and
        we can interpret sum nodes as latent Categorical variables.
    :param eps:
        Epsilon used by log semiring for numerical stability.
    :param compile:
        Wrap the module with :func:`torch.compile` (CUDA-graph capture via
        :code:`triton.cudagraphs`) so that repeated, same-shape evaluations replay as a 
        captured CUDA graph. This is typically several times faster on GPU, especially 
        for single-instance forward passes which are otherwise kernel-launch bound. 
        The first evaluation pays a one-time compilation cost, and compiling many structurally 
        different circuits in a single process can hit torch's recompilation limit 
        (pass :code:`compile=False` there).
    """
    import torch
    from .torch.circuit_modules import ProbabilisticCircuitModule, CircuitModule
    indices = self._get_indices()
    cls = ProbabilisticCircuitModule if probabilistic else CircuitModule
    module = cls(*indices, semiring=semiring, eps=eps)
    if compile:
        # triton.autotune_pointwise is disabled to reduce the cold-start cost.
        module = torch.compile(
            module, options={"triton.cudagraphs": True, "triton.autotune_pointwise": False}
        )
    return module


def to_jax_function(self: Circuit, semiring: str = "log", eps: float = 0):
    """
    Convert the circuit into a Jax function.

    :param semiring:
        The semiring in which the circuit should be evaluated. Supported options are :code:`"log"`, :code:`"real"`, :code:`"mpe"`, or :code:`"godel"`.
    :param eps:
        Epsilon used by log semiring for numerical stability.
    """
    from .jax import create_knowledge_layer
    indices = self._get_indices()
    return create_knowledge_layer(*indices, semiring=semiring, eps=eps)


def add_sdd(self: Circuit, sdd: "SddNode", true_lits: Sequence[int] = (), false_lits: Sequence[int] = ()) -> NodePtr:
    """
    Add an SDD to the Circuit.

    :param sdd:
        PySDD `SDDNode`_ to be added.
    :param true_lits:
        List of literals that are always true and should get propagated away.
    :param false_lits:
        List of literals that are always false and should get propagated away.

    .. _SDDNode: https://pysdd.readthedocs.io/en/latest/classes/SddNode.html
    """
    # Use delete=False for Windows compatibility - the file must be closed
    # before other processes can access it on Windows
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sdd.save(bytes(Path(tmp_path)))
        return self.add_sdd_from_file(tmp_path, true_lits, false_lits)
    finally:
        os.unlink(tmp_path)


Circuit.to_torch_module = to_torch_module
Circuit.to_jax_function = to_jax_function
Circuit.add_sdd = add_sdd
