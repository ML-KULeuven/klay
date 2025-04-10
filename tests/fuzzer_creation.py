from random import randint, choice, seed

import klay
from tqdm import tqdm


def sample_literal(circuit, max_depth, nb_vars: int = 100):
    lit = choice([-1, 1]) * randint(1, nb_vars)
    return circuit.literal_node(lit)


def sample_or_node(circuit, max_depth, max_nb_children: int = 6):
    nb_children = randint(0, max_nb_children)
    children = [sample_node(circuit, max_depth) for _ in range(nb_children)]
    return circuit.or_node(children)

def sample_and_node(circuit, max_depth, max_nb_children: int = 6):
    nb_children = randint(0, max_nb_children)
    children = [sample_node(circuit, max_depth) for _ in range(nb_children)]
    return circuit.and_node(children)


def sample_node(circuit, max_depth):
    if max_depth == 1:
        sample = sample_literal
    else:
        sample = choice([sample_literal, sample_or_node, sample_and_node])
    return sample(circuit, max_depth-1)


def sample_circuit():
    circuit = klay.Circuit()
    root = sample_or_node(circuit, 9)
    print(f"sampled circuit with {circuit.nb_nodes()} nodes")
    circuit.set_root(root)
    circuit.to_torch_module(semiring="real")
    print("done")


def fuzzer(nb_attempts=1000):
    for _ in tqdm(range(nb_attempts)):
        sample_circuit()

if __name__ == "__main__":
    seed(4)
    fuzzer()