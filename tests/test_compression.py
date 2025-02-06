import klay


def test_propagate_simple_true():
    c = klay.Circuit()
    t = c.true_node()
    f = c.false_node()
    l1, l2 = c.literal_node(1), c.literal_node(2)

    # test on input node
    assert c.nb_nodes() == 4
    and_node1 = c.and_node([t, l1]) # l1
    and_node2 = c.and_node([l1, t]) # l1
    or_node1 = c.or_node([l1, t]) # t
    or_node2 = c.or_node([t, l1]) # t
    assert c.nb_nodes() == 4

    # test on intermediate node
    l1_l2 = c.and_node([l1, l2])
    assert c.nb_nodes() == 5
    and_node1 = c.and_node([t, l1_l2]) # l1
    and_node2 = c.and_node([l1_l2, t]) # l1
    or_node1 = c.or_node([l1_l2, t]) # t
    or_node2 = c.or_node([t, l1_l2]) # t
    assert c.nb_nodes() == 5


def test_propagate_simple_false():
    c = klay.Circuit()
    t = c.true_node()
    f = c.false_node()
    l1, l2 = c.literal_node(1), c.literal_node(2)

    # test on input node
    assert c.nb_nodes() == 4
    and_node1 = c.and_node([f, l1])  # f
    and_node2 = c.and_node([l1, f])  # f
    or_node1 = c.or_node([l1, f])  # l1
    or_node2 = c.or_node([f, l1])  # l1
    assert c.nb_nodes() == 4

    # test on intermediate node
    l1_l2 = c.and_node([l1, l2])
    assert c.nb_nodes() == 5
    and_node1 = c.and_node([f, l1_l2])  # f
    and_node2 = c.and_node([l1_l2, f])  # f
    or_node1 = c.or_node([l1_l2, f])  # l1 & l2
    or_node2 = c.or_node([f, l1_l2])  # l1 & l2
    assert c.nb_nodes() == 5


def test_propagate_simple_ternary():
    """ test ternary nodes """
    c = klay.Circuit()
    t = c.true_node()
    f = c.false_node()
    l1, l2 = c.literal_node(1), c.literal_node(2)

    # test on true
    assert c.nb_nodes() == 4
    and_node1 = c.and_node([t, l1, l2])  # l1 & l2
    assert c.nb_nodes() == 5
    and_node2 = c.and_node([l2, t, l1])  # l1 & l2
    assert c.nb_nodes() == 5
    or_node1 = c.or_node([l1, t, l2])  # t
    assert c.nb_nodes() == 5
    or_node2 = c.or_node([l2, l1, t])  # t
    assert c.nb_nodes() == 5

    # test on false
    and_node3 = c.and_node([f, l1, l2])  # f
    assert c.nb_nodes() == 5
    and_node4 = c.and_node([l2, f, l1])  # f
    assert c.nb_nodes() == 5
    or_node3 = c.or_node([l1, f, l2])  # l1 | l2
    assert c.nb_nodes() == 8, "Expected 8 nodes instead of 6, because l1 and l2 require dummy nodes for the OR-node."
    or_node4 = c.or_node([l2, l1, f])  # l1 | l2
    assert c.nb_nodes() == 8


def test_removing_useless_nodes1():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(2), c.literal_node(3)
    assert c.nb_nodes() == 3
    and1 = c.and_node([l1, l2])
    assert c.nb_nodes() == 4
    or1 = c.or_node([and1, l3])
    assert c.nb_nodes() == 6  # or1 + 1 dummy node
    c.set_root(and1)
    # and1 is root node; but or1 is in a layer above, unused.
    assert c.nb_nodes() == 6
    c.remove_unused_nodes()  # should remove or1 + 1 dummy node
    assert c.nb_nodes() == 4, f"Expected 4 nodes instead of {c.nb_nodes()}"


def test_removing_useless_nodes2():
    c = klay.Circuit()
    l1, l2, l3 = c.literal_node(1), c.literal_node(2), c.literal_node(3)
    assert c.nb_nodes() == 3
    and1 = c.and_node([l1, l2])
    assert c.nb_nodes() == 4
    or1 = c.or_node([and1, l3])
    assert c.nb_nodes() == 6  # or1 + 1 dummy node
    and2 = c.and_node([l1, l3])  # useless
    assert c.nb_nodes() == 7
    or2 = c.or_node([l1, l2])  # useless
    assert c.nb_nodes() == 10  # or2 + 2 dummy nodes
    c.set_root(or1)
    c.remove_unused_nodes()  # should remove `and2`, `or2`, and 2 dummy nodes
    assert c.nb_nodes() == 6, f"Expected 5 nodes instead of {c.nb_nodes()}"
