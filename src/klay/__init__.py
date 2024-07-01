# noinspection PyUnresolvedReferences
from .nanobind_ext import Circuit

from .backends import torch_backend


def to_layered_module(circuit: Circuit):
    i1, i2 = circuit.get_indices()
    i1, i2 = i1[1:], i2[1:]
    return torch_backend.KnowledgeLayer(i1, i2)


Circuit.to_layered_module = to_layered_module
