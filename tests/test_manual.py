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
