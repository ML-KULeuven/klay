from .. import __lib
from .backends import torch_backend


def parse_tensors(file_name):
    import numpy as np

    with open(file_name, 'r') as f:
        lines = f.readlines()

    layers = []
    for line in lines:
        line = line.strip()
        if line.startswith("Layer"):
            layer = []
            layers.append(layer)
        else:
            layer.append([int(x) for x in line.split(" ")])
    layers = [np.array(layer) for layer in layers]

    # some sanity checks
    for i, layer in enumerate(layers):
        assert np.all(layer >= 0)
        if i != 0:
            assert np.all(layer < layers[i-1].shape[0])

    return layers


# This file contains python wrappers for our C++ functions.
# The whole purpose of that is to make it easier for
# auto-completions to know our function definitions.

# __lib is the compiled library containing our c++ functions.


def brr(name):
    return __lib.brr(name)
