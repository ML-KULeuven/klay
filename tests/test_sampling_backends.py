from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
springmv = pytest.importorskip("springmv")
torch = pytest.importorskip("torch")

from klay.torch.layers import ProbabilisticLogSumLayer


ROW_OFFSETS = np.array([0, 3, 5, 8, 10], dtype=np.int32)
COLUMN_INDICES = np.array(
    [0, 2, 4, 1, 3, 0, 3, 5, 2, 4], dtype=np.int32
)
NUM_COLS = 6


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("sampling backend comparison requires CUDA PyTorch")
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("sampling backend comparison requires CUDA JAX")
    if not devices or "cuda" not in devices[0].client.platform_version.lower():
        pytest.skip("sampling backend comparison requires CUDA JAX")


def _edge_rows_torch(device: torch.device) -> torch.Tensor:
    offsets = torch.as_tensor(ROW_OFFSETS, dtype=torch.long, device=device)
    return torch.repeat_interleave(
        torch.arange(ROW_OFFSETS.size - 1, device=device),
        torch.diff(offsets),
    )


def _torch_batched_sample(
    log_weights: torch.Tensor,
    gumbels: torch.Tensor,
    active_parent: torch.Tensor,
) -> torch.Tensor:
    """Batched form of Klay's segment-max and scatter-sum sampler."""
    batch_size = gumbels.shape[0]
    device = gumbels.device
    offsets = torch.as_tensor(ROW_OFFSETS, dtype=torch.long, device=device)
    columns = torch.as_tensor(
        COLUMN_INDICES, dtype=torch.long, device=device
    )
    edge_rows = _edge_rows_torch(device)
    scores = gumbels + (
        log_weights if log_weights.dim() == 2 else log_weights.unsqueeze(0)
    )
    edge_major_scores = scores.mT.contiguous()
    row_max = torch.segment_reduce(
        edge_major_scores,
        "max",
        offsets=offsets,
        axis=0,
        unsafe=True,
    )
    selected = edge_major_scores == row_max[edge_rows]
    selected &= active_parent.mT[edge_rows]
    counts = torch.zeros(
        (NUM_COLS, batch_size), dtype=torch.int32, device=device
    )
    child_indices = columns[:, None].expand_as(selected)
    active_child = torch.scatter_reduce(
        counts,
        0,
        index=child_indices,
        src=selected.to(torch.int32),
        reduce="sum",
        include_self=False,
    )
    return (active_child > 0).mT


def test_springmv_matches_klay_torch_layer_with_same_noise() -> None:
    _require_cuda()
    device = torch.device("cuda")
    offsets = torch.as_tensor(ROW_OFFSETS, dtype=torch.long, device=device)
    columns = torch.as_tensor(
        COLUMN_INDICES, dtype=torch.long, device=device
    )
    layer = ProbabilisticLogSumLayer(
        columns,
        offsets,
        _edge_rows_torch(device),
        eps=0.0,
    ).to(device)
    raw_weights = torch.tensor(
        [0.2, -0.4, 0.7, 1.1, -0.2, 0.3, -0.8, 0.6, 0.9, -0.5],
        dtype=torch.float32,
        device=device,
    )
    uniform = torch.tensor(
        [0.13, 0.79, 0.37, 0.62, 0.21, 0.91, 0.48, 0.33, 0.72, 0.56],
        dtype=torch.float32,
        device=device,
    )
    active_parent = torch.tensor(
        [True, False, True, True], device=device
    )
    with torch.no_grad():
        layer.weights.copy_(raw_weights)
        log_weights = layer.get_log_edge_weights()
        gumbels = -torch.log(-torch.log(uniform))
        with mock.patch.object(torch, "rand_like", return_value=uniform):
            torch_result = layer.sample(active_parent)

    structure = springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=NUM_COLS
    )
    springmv_result = jax.jit(springmv.csr_logsumexp_sample)(
        structure,
        jnp.asarray(log_weights.detach().cpu().numpy()),
        jnp.asarray(gumbels.cpu().numpy()),
        jnp.asarray(active_parent.cpu().numpy()),
    )
    np.testing.assert_array_equal(
        np.asarray(springmv_result), torch_result.cpu().numpy()
    )


def test_batched_springmv_matches_vectorized_torch_sampler() -> None:
    _require_cuda()
    rng = np.random.default_rng(7)
    batch_size = 8
    weights = rng.normal(size=(batch_size, COLUMN_INDICES.size)).astype(
        np.float32
    )
    uniform = rng.uniform(
        1.0e-6, 1.0 - 1.0e-6, size=(batch_size, COLUMN_INDICES.size)
    ).astype(np.float32)
    gumbels = -np.log(-np.log(uniform)).astype(np.float32)
    active_parent = rng.random((batch_size, ROW_OFFSETS.size - 1)) < 0.5

    device = torch.device("cuda")
    torch_result = _torch_batched_sample(
        torch.as_tensor(weights, device=device),
        torch.as_tensor(gumbels, device=device),
        torch.as_tensor(active_parent, device=device),
    )
    structure = springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=NUM_COLS
    )
    springmv_operation = jax.jit(
        jax.vmap(
            lambda current_weights, current_gumbels, current_active: (
                springmv.csr_logsumexp_sample(
                    structure,
                    current_weights,
                    current_gumbels,
                    current_active,
                )
            )
        )
    )
    springmv_result = springmv_operation(weights, gumbels, active_parent)
    np.testing.assert_array_equal(
        np.asarray(springmv_result), torch_result.cpu().numpy()
    )
