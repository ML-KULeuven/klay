import klay as k

def test_full_pipeline_no_exception():
    c = k.Circuit()
    a = c.literal_node(1)
    b = c.literal_node(-2)
    d = c.literal_node(3)
    and1 = c.and_node([a, b])
    and2 = c.and_node([c.literal_node(-1), b])
    or1  = c.or_node([and1, and2])
    and3 = c.and_node([or1, d])
    c.set_root(and3)
    c.set_root(and1)
    r = k.check_sdnnf(c)
    assert isinstance(r, k.SDNNFResult)
    indices, csr = c._get_indices()
    assert len(indices) == len(csr)


def test_result_type_integrity():
    c = k.Circuit()
    c.set_root(c.literal_node(1))
    r = k.check_sdnnf(c)
    assert isinstance(r.is_nnf,          bool)
    assert isinstance(r.is_decomposable, bool)
    assert isinstance(r.is_smooth,       bool)
    assert isinstance(r.is_dnnf,         bool)
    assert isinstance(r.is_sdnnf,        bool)
    assert isinstance(r.n_and,           int)
    assert isinstance(r.n_or,            int)
    assert isinstance(r.n_vars_found,    int)
    assert isinstance(r.violations,      list)
    assert isinstance(r.summary(),       str)


def test_flat_and():
    c = k.Circuit()
    n_vars = 5
    lits = [c.literal_node(i) for i in range(1, n_vars + 1)]
    c.set_root(c.and_node(lits))
    r = k.check_sdnnf(c)
    assert r.is_decomposable,   r.summary()
    assert r.is_smooth,         r.summary()
    assert r.n_vars_found == n_vars
    assert len(r.violations) == 0


def test_large_smooth_or():
    n_vars = 100
    c = k.Circuit()
    children = []
    for _ in range(10):
        lits  = [c.literal_node((j + 1)) for j in range(n_vars)]
        children.append(c.and_node(lits))
    c.set_root(c.or_node(children))
    r = k.check_sdnnf(c)
    assert r.is_smooth,         r.summary()
    assert r.is_decomposable,   r.summary()
    assert r.n_vars_found == n_vars
    assert len(r.violations) == 0


def test_violates_decomposability():
    c = k.Circuit()
    c.set_root(c.and_node([c.literal_node(5), c.literal_node(-5)]))
    r = k.check_sdnnf(c)
    assert r.is_smooth,             r.summary()
    assert not r.is_decomposable,   r.summary()
    assert len(r.violations) == 1
    assert any(v.property == "decomposability" for v in r.violations)


def test_chain_shared_var_depth():
    c = k.Circuit()
    x1 = c.literal_node(1)
    x2 = c.literal_node(2)
    node = c.and_node([x1, x2])
    for _ in range(29):
        node = c.and_node([x1, node])
    c.set_root(node)
    r = k.check_sdnnf(c)
    assert r.is_smooth,             r.summary()
    assert not r.is_decomposable,   r.summary()
    assert len(r.violations) == 29
    assert all(v.property == "decomposability" for v in r.violations)


def test_or_children_differ_by():
    c = k.Circuit()
    full  = c.and_node([c.literal_node(i)  for i in range(1, 11)])
    short = c.and_node([c.literal_node(i)  for i in range(1, 10)])
    c.set_root(c.or_node([full, short]))
    r = k.check_sdnnf(c)
    assert r.is_decomposable,   r.summary()
    assert not r.is_smooth,     r.summary()
    assert any(v.property == "smoothness" for v in r.violations)
    assert "10" in r.violations[0].detail


def test_or_of_single_literals_not_smooth():
    c = k.Circuit()
    lits = [c.literal_node(i) for i in range(1, 51)]
    c.set_root(c.or_node(lits))
    r = k.check_sdnnf(c)
    assert r.is_decomposable,   r.summary()
    assert not r.is_smooth,     r.summary()
    assert any(v.property == "smoothness" for v in r.violations)


def test_decomposable_but_not_smooth():
    c = k.Circuit()
    x1 = c.literal_node(1)
    x2 = c.literal_node(2)
    and12 = c.and_node([x1, x2])
    c.set_root(c.or_node([and12, c.literal_node(1)]))
    r = k.check_sdnnf(c)
    assert r.is_decomposable, r.summary()
    assert not r.is_smooth,   r.summary()
    assert r.is_dnnf,         r.summary()
    assert not r.is_sdnnf,    r.summary()
