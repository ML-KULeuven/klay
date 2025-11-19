"""
Test for numerical stability issues in backward pass with log probabilities.

This test identifies cases where forward pass produces finite values but
backward pass introduces NaNs, particularly when inputs are close to -inf
or when log probabilities approach 0 (probability = 1).
"""
import torch

from klay.torch.utils import log1mexp


def test_log1mexp_gradient_stability():
    test_cases = [-1e-10, -0.01, -0.1, -1.0, -10.0, -100.0, -1000.0]

    for x in test_cases:
        x = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
        out = log1mexp(x)

        assert torch.isfinite(out), f"Output is not finite for {x}."
        out.backward()
        assert torch.isfinite(x.grad), f"Gradient is not finite for {x}."


if __name__ == "__main__":
    test_log1mexp_gradient_stability()
