import torch
import numpy as np


def parse_tensors(file_name):
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
    return layers


def encode_input(pos, neg=None):
    if neg is None:
        neg = 1 - pos

    shape = (2 * x.shape[0] + 2,) + x.shape[1:]
    result = torch.empty(shape)
    result[2::2] = pos
    result[3::2] = neg
    result[0] = 0
    result[1] = 1
    return result


class KnowledgeLayer(torch.nn.Module):
    def __init__(self, layers):
        super(KnowledgeLayer, self).__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            if i % 2 == 0:
                self.layers.append(ProductLayer(layer))
            else:
                self.layers.append(SumLayer(layer))
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = encode_input(x)
        print("INPUT")
        print(x)
        return self.layers(x)


class SumLayer(torch.nn.Module):
    def __init__(self, indices):
        super(SumLayer, self).__init__()
        self.indices = indices

    def forward(self, x):
        print("SUM")
        result = x[self.indices].sum(axis=1)
        print(result)
        return result


class ProductLayer(torch.nn.Module):
    def __init__(self, indices):
        super(ProductLayer, self).__init__()
        self.indices = indices

    def forward(self, x):
        print("PROD")
        result = x[self.indices].prod(axis=1)
        print(result)
        return result


if __name__ == "__main__":
    tensors = parse_tensors("tensors.txt")
    kl = KnowledgeLayer(tensors)
    x = torch.tensor([.5, .4, .3, .2, .1, .0, .9, .8, .7, .6])
    print(kl(x))

