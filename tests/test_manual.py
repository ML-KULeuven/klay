import klay
import torch
from pysdd.sdd import SddManager


def test_node_equality():
    c = klay.Circuit()
    assert c.true_node() == c.true_node()
    assert c.true_node() != c.false_node()


def test_or_node():
    c = klay.Circuit()
    l1, l2 = c.literal_node(1), c.literal_node(-2)
    c.set_root(c.or_node([l1, l2]))

    m = c.to_torch_module(semiring='real')
    weights = torch.tensor([0.4, 0.8])
    assert m(weights) == 0.4 + (1 - 0.8)


def test_probabilistic():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(-2), c.literal_node(3)
    or_node1 = c.or_node([l1, l2])
    or_node2 = c.or_node([l2, l3])
    and_node = c.and_node([or_node1, or_node2])
    c.set_root(and_node)

    m = c.to_torch_module(semiring='real', probabilistic=True)
    m.layers[1].weights.data.zero_()
    weights = torch.tensor([0.4, 0.8, 0.5])
    expected_result = torch.tensor((0.4/2 + 0.2/2) * (0.2/2 + 0.5/2))
    assert torch.allclose(m(weights), expected_result)

def test_log_probabilistic():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(-2), c.literal_node(3)
    or_node1 = c.or_node([l1, l2])
    or_node2 = c.or_node([l2, l3])
    and_node = c.and_node([or_node1, or_node2])
    c.set_root(and_node)

    m = c.to_torch_module(semiring='log', probabilistic=True)
    m.layers[1].weights.data.zero_()
    weights = torch.tensor([0.4, 0.8, 0.5])
    expected_result = torch.tensor((0.4/2 + 0.2/2) * (0.2/2 + 0.5/2))
    assert torch.allclose(m(weights.log()).exp(), expected_result)


def test_multi_rooted():
    c = klay.Circuit()
    l1, l2 = c.literal_node(1), c.literal_node(-2)
    c.set_root(c.or_node([l1, l2]))
    c.set_root(c.and_node([l1, l2]))

    m = c.to_torch_module(semiring='real')
    weights = torch.tensor([0.4, 0.8])
    expected = torch.tensor([0.4 + 0.2, 0.4 * 0.2])
    assert torch.allclose(m(weights), expected)


def test_multi_rooted2():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(2), c.literal_node(3)
    and1 = c.and_node([l1, l2])
    and2 = c.and_node([l2, l3])
    c.set_root(and1)
    c.set_root(and2)

    m = c.to_torch_module(semiring='real')
    w = torch.tensor([0.4, 0.8, 0.6])
    expected = torch.tensor([0.4 * 0.8, 0.8 * 0.6])
    assert torch.allclose(m(w), expected)


def test_multi_rooted_ordering():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(2), c.literal_node(3)
    and1 = c.and_node([l1, l2])
    and2 = c.and_node([l2, l3])
    c.set_root(and2)
    c.set_root(and1)

    m = c.to_torch_module(semiring='real')
    w = torch.tensor([0.4, 0.8, 0.6])
    expected = torch.tensor([0.8 * 0.6, 0.4 * 0.8])
    print(m(w), expected)
    assert torch.allclose(m(w), expected)


def test_single_layer_multi_root():
    c = klay.Circuit()
    l1, l2 = c.literal_node(1), c.literal_node(-2)
    c.set_root(l1)
    c.set_root(l2)
    c.set_root(l1)

    m = c.to_torch_module(semiring='real')
    weights = torch.tensor([0.4, 0.8])
    expected = torch.tensor([0.4, 0.2, 0.4])
    assert torch.allclose(m(weights), expected)

def test_superfluous_nodes_after_root():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(2), c.literal_node(3)
    and_node = c.and_node([l1, l2])
    or_node = c.or_node([and_node, l3])
    c.set_root(and_node)

    weights = torch.tensor([0.25, 0.5, 0.2])
    expected = torch.tensor([0.125])
    m = c.to_torch_module(semiring='real')
    assert torch.allclose(m(weights), expected)


def test_sdd_literal():
    sdd_mgr = SddManager(var_count=2)
    a, b = sdd_mgr.vars

    c = klay.Circuit()
    c.add_sdd(a)
    m = c.to_torch_module(semiring='real')
    weights = torch.tensor([0.4])
    expected = torch.tensor([0.4])
    assert torch.allclose(m(weights), expected)

def test_sdd_multiroot():
    sdd_mgr = SddManager(var_count=2)
    a, b = sdd_mgr.vars

    c = klay.Circuit()
    c.add_sdd(a)
    c.add_sdd(a & b)
    c.add_sdd(a & b & b)
    c.add_sdd(a & a)
    m = c.to_torch_module(semiring='real')
    weights = torch.tensor([0.2, 0.5])
    expected = torch.tensor([0.2, 0.1, 0.1, 0.2])
    assert torch.allclose(m(weights), expected)