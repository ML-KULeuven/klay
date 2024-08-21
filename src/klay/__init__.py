# noinspection PyUnresolvedReferences
from .nanobind_ext import Circuit

from .backends import torch_backend


def to_torch_module(circuit: Circuit):
    indices = circuit.get_indices()
    return torch_backend.KnowledgeLayer(*indices)


def add_sdd(circuit: Circuit, sdd: "SddNode"):
    import os
    from pathlib import Path

    sdd.save(bytes(Path("tmp.sdd")))
    circuit.add_SDD_from_file("tmp.sdd")
    os.remove("tmp.sdd")


Circuit.to_torch_module = to_torch_module
Circuit.add_sdd = add_sdd
