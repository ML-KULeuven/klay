import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import klay


def _circuit():
    circuit = klay.Circuit()
    positive_one = circuit.literal_node(1)
    negative_one = circuit.literal_node(-1)
    positive_two = circuit.literal_node(2)
    left = circuit.and_node([positive_one, positive_two])
    right = circuit.and_node([negative_one, positive_two])
    circuit.set_root(circuit.or_node([left, right]))
    return circuit


def _require_springmv_cuda():
    pytest.importorskip("springmv")
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("SpringMV integration requires an NVIDIA CUDA device")
    if not devices or "cuda" not in devices[0].client.platform_version.lower():
        pytest.skip("SpringMV integration requires an NVIDIA CUDA device")


@pytest.mark.parametrize("semiring", ["log", "godel"])
def test_springmv_matches_jax(semiring):
    _require_springmv_cuda()
    native = _circuit().to_jax_function(semiring=semiring, backend="jax")
    springmv = _circuit().to_jax_function(
        semiring=semiring,
        backend="springmv",
    )
    inputs = jnp.array([-0.7, -0.2], dtype=jnp.float32)
    np.testing.assert_allclose(springmv(inputs), native(inputs), rtol=1e-6)


def test_springmv_log_vmap_and_gradient_match_jax():
    _require_springmv_cuda()
    native = _circuit().to_jax_function(semiring="log", backend="jax")
    springmv = _circuit().to_jax_function(
        semiring="log",
        backend="springmv",
    )
    inputs = jnp.array(
        [[-0.7, -0.2], [-1.2, -0.4], [-0.3, -1.1]],
        dtype=jnp.float32,
    )
    native_batched = jax.vmap(native)(inputs)
    springmv_batched = jax.vmap(springmv)(inputs)
    np.testing.assert_allclose(
        springmv_batched,
        native_batched,
        rtol=1e-6,
    )

    point = inputs[0]
    native_gradient = jax.grad(lambda x: native(x).sum())(point)
    springmv_gradient = jax.grad(lambda x: springmv(x).sum())(point)
    np.testing.assert_allclose(
        springmv_gradient,
        native_gradient,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("semiring", ["real", "mpe"])
def test_springmv_rejects_product_reductions(semiring):
    with pytest.raises(NotImplementedError, match="product reduction"):
        _circuit().to_jax_function(semiring=semiring, backend="springmv")


def test_springmv_rejects_nonzero_log_epsilon():
    with pytest.raises(NotImplementedError, match="nonzero eps"):
        _circuit().to_jax_function(
            semiring="log",
            eps=1e-6,
            backend="springmv",
        )


def test_jax_rejects_unknown_sparse_backend():
    with pytest.raises(ValueError, match="Unknown JAX sparse backend"):
        _circuit().to_jax_function(backend="unknown")
