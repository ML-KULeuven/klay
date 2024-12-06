import klay
import torch


def test_node_equality():
    c = klay.Circuit()
    assert c.true_node() == c.true_node()
    assert c.true_node() != c.false_node()


def test_or_node():
    c = klay.Circuit()
    l1, l2 = c.literal_node(1), c.literal_node(-2)
    or_node = c.or_node([l1, l2])

    m = c.to_torch_module(semiring='real')
    weights = torch.tensor([0.4, 0.8])
    assert m(weights) == 0.4 + (1 - 0.8)


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
