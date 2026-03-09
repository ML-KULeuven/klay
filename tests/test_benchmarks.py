"""Benchmarks for circuit evaluation. Run with: uv run pytest tests/test_benchmarks.py --benchmark-only"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("pysdd")

import torch
import klay
from pysdd.sdd import SddManager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_circuit():
    """Small circuit: 10 variables."""
    mgr = SddManager(var_count=10)
    vs = list(mgr.vars)
    sdd = vs[0] & vs[1]
    for v in vs[2:]:
        sdd = sdd | (sdd & v)
    c = klay.Circuit()
    c.add_sdd(sdd)
    return c, 10


@pytest.fixture(scope="module")
def medium_circuit():
    """Medium circuit: 50 variables."""
    mgr = SddManager(var_count=50)
    vs = list(mgr.vars)
    sdd = vs[0] & vs[1]
    for v in vs[2:]:
        sdd = sdd | (sdd & v)
    c = klay.Circuit()
    c.add_sdd(sdd)
    return c, 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _forward(module, weights):
    module(weights)


def _forward_backward(module, weights):
    w = weights.detach().requires_grad_(True)
    module(w).sum().backward()


# ── Non-probabilistic benchmarks ─────────────────────────────────────────────

@pytest.mark.parametrize("semiring", ["real", "log", "mpe", "godel"])
def test_forward_small(benchmark, small_circuit, semiring):
    circuit, nb_vars = small_circuit
    m = circuit.to_torch_module(semiring=semiring)
    weights = torch.rand(nb_vars)
    benchmark(_forward, m, weights)


@pytest.mark.parametrize("semiring", ["real", "log", "mpe", "godel"])
def test_forward_backward_small(benchmark, small_circuit, semiring):
    circuit, nb_vars = small_circuit
    m = circuit.to_torch_module(semiring=semiring)
    weights = torch.rand(nb_vars)
    benchmark(_forward_backward, m, weights)


@pytest.mark.parametrize("semiring", ["real", "log", "mpe", "godel"])
def test_forward_medium(benchmark, medium_circuit, semiring):
    circuit, nb_vars = medium_circuit
    m = circuit.to_torch_module(semiring=semiring)
    weights = torch.rand(nb_vars)
    benchmark(_forward, m, weights)


@pytest.mark.parametrize("semiring", ["real", "log", "mpe", "godel"])
def test_forward_backward_medium(benchmark, medium_circuit, semiring):
    circuit, nb_vars = medium_circuit
    m = circuit.to_torch_module(semiring=semiring)
    weights = torch.rand(nb_vars)
    benchmark(_forward_backward, m, weights)


# ── Probabilistic benchmarks ─────────────────────────────────────────────────

@pytest.mark.parametrize("semiring", ["real", "log"])
def test_prob_forward_medium(benchmark, medium_circuit, semiring):
    circuit, nb_vars = medium_circuit
    m = circuit.to_torch_module(semiring=semiring, probabilistic=True)
    weights = torch.rand(nb_vars)
    if semiring == "log":
        weights = weights.log()
    benchmark(_forward, m, weights)


@pytest.mark.parametrize("semiring", ["real", "log"])
def test_prob_forward_backward_medium(benchmark, medium_circuit, semiring):
    circuit, nb_vars = medium_circuit
    m = circuit.to_torch_module(semiring=semiring, probabilistic=True)
    weights = torch.rand(nb_vars)
    if semiring == "log":
        weights = weights.log()
    benchmark(_forward_backward, m, weights)


# ── Sampling benchmark ───────────────────────────────────────────────────────

def test_prob_sample_medium(benchmark, medium_circuit):
    circuit, nb_vars = medium_circuit
    m = circuit.to_torch_module(semiring='real', probabilistic=True)
    weights = torch.rand(nb_vars)
    m(weights)  # initialize
    benchmark(m.sample)
