"""
Test for numerical stability issues in backward pass with log probabilities.

This test identifies cases where forward pass produces finite values but
backward pass introduces NaNs, particularly when inputs are close to -inf
or when log probabilities approach 0 (probability = 1).
"""
import torch
from klay.torch.utils import log1mexp


def test_log1mexp_gradient_stability():
	test_cases = [
		torch.tensor([-0.01, -0.1, -1.0, -10.0, -100.0], dtype=torch.float32),
		torch.tensor([-1000.0], dtype=torch.float32),
		torch.tensor([-1e-10], dtype=torch.float32),  # Very close to 0
	]

	for i, x in enumerate(test_cases):
		x_test = x.clone().requires_grad_(True)


		output = log1mexp(x_test)

		assert torch.isfinite(output).all(), f"Output should be finite for case {i + 1}"

		loss = output.sum()
		loss.backward()

		assert torch.isfinite(x_test.grad).all(), f"Gradient should be finite for case {i + 1}"
