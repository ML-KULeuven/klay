# noinspection PyUnresolvedReferences
from .nanobind_ext import Circuit


def to_torch_module(circuit: Circuit, semiring: str = "log"):
    from .backends import torch_backend
    indices = circuit._get_indices()
    return torch_backend.KnowledgeModule(*indices, semiring=semiring)


def to_jax_function(circuit: Circuit, semiring: str = "log"):
    from .backends import jax_backend
    indices = circuit._get_indices()
    return jax_backend.create_knowledge_layer(*indices, semiring=semiring)


def add_sdd(circuit: Circuit, sdd: "SddNode", **kwargs):
    import os
    from pathlib import Path

    sdd.save(bytes(Path("tmp.sdd")))
    circuit.add_SDD_from_file("tmp.sdd", **kwargs)
    os.remove("tmp.sdd")


Circuit.to_torch_module = to_torch_module
Circuit.to_jax_function = to_jax_function
Circuit.add_sdd = add_sdd
