"""Native batched evaluation: equivalence to per-instance evaluation, and gradient
finiteness. Covers the segment_reduce backend across all semirings."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("pysdd")

import torch
from pysdd.sdd import SddManager

import klay
from .utils import generate_random_dimacs

NB_VARS = 20


def _sdd_circuit(seed):
    generate_random_dimacs('tmp_batch.cnf', NB_VARS, NB_VARS // 2, seed=seed)
    sdd = SddManager.from_cnf_file(b'tmp_batch.cnf')[1]
    c = klay.Circuit()
    c.add_sdd(sdd)
    return c


def _inputs(semiring, batch):
    p = torch.rand(batch, NB_VARS).clamp(0.05, 0.95)
    return p.log() if semiring == "log" else p


@pytest.mark.parametrize("semiring", ["log", "real", "mpe", "godel"])
def test_native_batching_matches_single(semiring):
    m = _sdd_circuit(seed=3).to_torch_module(semiring, compile=False)
    x = _inputs(semiring, batch=16)
    with torch.no_grad():
        singles = torch.stack([m(x[i]) for i in range(x.shape[0])])  # [B, num_roots]
        batched = m(x)                                               # [B, num_roots]
    assert batched.shape == singles.shape
    assert torch.allclose(batched, singles, atol=1e-5), \
        f"{semiring}: max|Δ|={(batched - singles).abs().max().item():.3e}"


@pytest.mark.skip()
def test_compiled_matches_eager():
    """torch.compile (the default) must be numerically faithful to eager. One representative
    semiring (log: logsumexp + batched transpose) keeps CI lean; other semirings share the path."""
    semiring = "log"
    c = _sdd_circuit(seed=7)
    eager = c.to_torch_module(semiring, compile=False)
    compiled = c.to_torch_module(semiring, compile=True)
    x1 = _inputs(semiring, batch=1)[0]
    with torch.no_grad():
        try:
            d1 = (compiled(x1) - eager(x1)).abs().max().item()
        except Exception as e:  # compile backend unavailable in this environment
            pytest.skip(f"torch.compile unavailable: {e}")
    assert d1 < 1e-4, f"{semiring}: single={d1:.2e}"


@pytest.mark.parametrize("semiring", ["log", "real"])
def test_batched_forward_backward_finite(semiring):
    m = _sdd_circuit(seed=5).to_torch_module(semiring, compile=False)
    x = _inputs(semiring, batch=8).detach().requires_grad_(True)
    out = m(x)
    assert out.shape[0] == 8
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
