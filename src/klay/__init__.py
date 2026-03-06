# noinspection PyUnresolvedReferences
from .klay_ext import Circuit, NodePtr
NodePtr.__module__ = "klay"

from collections.abc import Sequence
import tempfile
import os
from pathlib import Path


def to_torch_module(self: Circuit, semiring: str = "log", probabilistic: bool = False, eps: float = 0):
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
    """
    from .torch.circuit_modules import ProbabilisticCircuitModule
    from .torch.circuit_modules import CircuitModule
    indices = self._get_indices()
    if probabilistic:
        return ProbabilisticCircuitModule(*indices, semiring=semiring, eps=eps)
    return CircuitModule(*indices, semiring=semiring, eps=eps)


def to_jax_function(self: Circuit, semiring: str = "log"):
    """
    Convert the circuit into a Jax function.

    :param semiring:
        The semiring in which the circuit should be evaluated. Supported options are :code:`"log"`, :code:`"real"`, :code:`"mpe"`, or :code:`"godel"`.
    """
    from .jax import create_knowledge_layer
    indices = self._get_indices()
    return create_knowledge_layer(*indices, semiring=semiring)


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
