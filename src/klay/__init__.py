# noinspection PyUnresolvedReferences
from .nanobind_ext import Circuit

from collections.abc import Sequence


def to_torch_module(self: Circuit, semiring: str = "log", probabilistic: bool = False):
    """
    Convert the circuit into a PyTorch module.

    :param semiring:
        The semiring in which the circuit should be evaluated. Supported options are ("log", "real", "mpe", "godel").
    :param probabilistic:
        If true, construct a probabilistic circuit instead of an arithmetic circuit.
        This means the inputs to a sum node are multiplied by a probability, and
        we can interpret sum nodes as latent Categorical variables.
    """
    from .backends import torch_backend
    indices = self._get_indices()
    return torch_backend.KnowledgeModule(*indices, semiring=semiring, probabilistic=probabilistic)


def to_jax_function(self: Circuit, semiring: str = "log"):
    """
    Convert the circuit into a Jax function.

    :param semiring:
        The semiring in which the circuit should be evaluated. Supported options are ("log", "real").
    """
    from .backends import jax_backend
    indices = self._get_indices()
    return jax_backend.create_knowledge_layer(*indices, semiring=semiring)


def add_sdd(self: Circuit, sdd: "SddNode", true_lits: Sequence[int] = (), false_lits: Sequence[int] = ()):
    """
    Add an SDD to the Circuit.
    """
    import os
    from pathlib import Path

    sdd.save(bytes(Path("tmp.sdd")))
    self.add_SDD_from_file("tmp.sdd", true_lits, false_lits)
    os.remove("tmp.sdd")


Circuit.to_torch_module = to_torch_module
Circuit.to_jax_function = to_jax_function
Circuit.add_sdd = add_sdd
