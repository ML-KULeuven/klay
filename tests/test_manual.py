import pytest

from klay.torch import ProbabilisticCircuitModule

pytest.importorskip("torch")
pytest.importorskip("pysdd")

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
    expected_result = torch.tensor((0.4 / 2 + 0.2 / 2) * (0.2 / 2 + 0.5 / 2))
    assert torch.allclose(m(weights), expected_result)


def test_create_pc():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(-2), c.literal_node(3)
    or_node1 = c.or_node([l1, l2])
    or_node2 = c.or_node([l2, l3])
    and_node = c.and_node([or_node1, or_node2])
    c.set_root(and_node)

    m = c.to_torch_module(semiring='real')
    m = ProbabilisticCircuitModule.from_circuit(m, torch.tensor([0.4, 0.8, 0.5]))
    edge_weights = m.layers[1].get_edge_weights()
    expected_weights = torch.tensor([2/3, 1/3, 2/7, 5/7])
    assert torch.allclose(edge_weights, expected_weights)


def test_pc_conditioning():
    c = klay.Circuit()
    p1, p2 = c.literal_node(1), c.literal_node(2)
    n1, n2 = c.literal_node(-1), c.literal_node(-2)
    and_node1 = c.and_node([p1, p2])
    and_node2 = c.and_node([n1, n2])
    or_node = c.or_node([and_node1, and_node2])
    c.set_root(or_node)

    m = c.to_torch_module(semiring='real', probabilistic=True)
    m.condition(torch.tensor([1, 1]), torch.tensor([1, 0]))
    for _ in range(20):
        assert torch.allclose(m.sample(), torch.tensor([True, True]))


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
    expected_result = torch.tensor((0.4 / 2 + 0.2 / 2) * (0.2 / 2 + 0.5 / 2))
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


def test_custom_semiring_tropical():
    """Test CircuitModule with a manually defined tropical (min-plus) semiring.

    Tropical semiring: ⊕ = min, ⊗ = +, zero = +∞, one = 0.
    For circuit AND(OR(l1, l2), OR(l2, l3)) with costs [1, 2, 3]:
      OR(l1, l2) = min(1, 2) = 1
      OR(l2, l3) = min(2, 3) = 2
      AND(...)   = 1 + 2     = 3
    """
    def tropical_negate(x):
        return -x

    tropical_semiring = ("amin", "sum", float('inf'), 0.0, tropical_negate)

    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(2), c.literal_node(3)
    or1 = c.or_node([l1, l2])
    or2 = c.or_node([l2, l3])
    c.set_root(c.and_node([or1, or2]))

    m = c.to_torch_module(semiring=tropical_semiring)
    costs = torch.tensor([1.0, 2.0, 3.0])
    result = m(costs)

    expected = torch.tensor([3.0])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_custom_callable_layer():
    """GatherCircuitLayer accepts a standard reduction + fill_value."""
    from klay.torch.layers import GatherCircuitLayer

    # Three groups of unequal size: [0,1,2] -> 0, [3,4] -> 1, [5] -> 2
    ix_in  = torch.tensor([0, 1, 2, 3, 4, 5])
    ix_out = torch.tensor([0, 0, 0, 1, 1, 2])
    layer = GatherCircuitLayer(ix_in, ix_out, reduce_fn=torch.nanmean, fill_value=float('nan'))

    x = torch.tensor([1.0, 3.0, 8.0, 2.0, 6.0, 5.0])
    result = layer(x)
    expected = torch.tensor([4.0, 4.0, 5.0])  # mean([1,3,8])=4, mean([2,6])=4, mean([5])=5
    assert torch.allclose(result, expected)


def test_custom_callable_semiring():
    """CircuitModule accepts a semiring with a custom callable reduction."""
    from klay.torch.utils import log1mexp

    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(-2), c.literal_node(3)
    or1 = c.or_node([l1, l2])
    or2 = c.or_node([l2, l3])
    c.set_root(c.and_node([or1, or2]))

    # Custom callable (torch.logsumexp) for sum, string for prod
    semiring = (torch.logsumexp, "sum", float('-inf'), 0.0, log1mexp)
    m = c.to_torch_module(semiring=semiring)

    weights = torch.tensor([0.4, 0.8, 0.5])
    expected = c.to_torch_module(semiring='log')(weights.log()).exp()
    assert torch.allclose(m(weights.log()).exp(), expected)


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
