# noinspection PyUnresolvedReferences
from .nanobind_ext import Circuit


def to_torch_module(circuit: Circuit):
    from .backends import torch_backend
    indices = circuit.get_indices()
    return torch_backend.KnowledgeLayer(*indices)


def to_jax_function(circuit: Circuit):
    from .backends import jax_backend
    indices = circuit.get_indices()
    return jax_backend.create_knowledge_layer(*indices)


def add_sdd(circuit: Circuit, sdd: "SddNode", **kwargs):
    import os
    from pathlib import Path

    sdd.save(bytes(Path("tmp.sdd")))
    circuit.add_SDD_from_file("tmp.sdd", **kwargs)
    os.remove("tmp.sdd")


Circuit.to_torch_module = to_torch_module
Circuit.to_jax_function = to_jax_function
Circuit.add_sdd = add_sdd
