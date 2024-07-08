# noinspection PyUnresolvedReferences
from .nanobind_ext import Circuit

from .backends import torch_backend


def to_layered_module(circuit: Circuit):
    indices = circuit.get_indices()
    return torch_backend.KnowledgeLayer(*indices)


Circuit.to_layered_module = to_layered_module
