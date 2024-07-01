# noinspection PyUnresolvedReferences
from .nanobind_ext import Circuit

from .backends import torch_backend


def parse_tensors(file_name):
    import numpy as np

    with open(file_name, 'r') as f:
        lines = f.readlines()

    layers = []
    for line in lines:
        line = line.strip()
        layers.append([int(x) for x in line.split(" ")])
    layers = [np.array(layer) for layer in layers]
    pointers = layers[1::2]
    csrs = layers[::2]

    return pointers, csrs
