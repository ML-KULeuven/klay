import argparse
from time import perf_counter

import numpy as np
from cirkit.templates.logic.sdd import SDD
from cirkit.pipeline import PipelineContext
from cirkit.backend.torch.queries import IntegrateQuery
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import CategoricalLayer
from cirkit.symbolic.parameters import Parameter, TensorParameter
from cirkit.utils.scope import Scope

import torch
from cirkit.backend.torch.layers.input import TorchCategoricalLayer

from .torch import _torch_weights

# cirkit's log_partition_function expects logits shaped (F, K, D, N) but
# TensorParameter(1, 2) produces (F, K, N). Patch to handle both.
def _log_partition_function(self):
    if self.logits is None:
        return torch.zeros(size=(self.num_folds, 1, self.num_output_units), device=self.probs.device)
    logits = self.logits()
    if logits.dim() == 3:
        logits = logits.unsqueeze(2)
    return torch.sum(torch.logsumexp(logits, dim=3), dim=2).unsqueeze(dim=1)

TorchCategoricalLayer.log_partition_function = _log_partition_function

CIRCUITS = ["sudoku_4", "4-grid", "seq_fun", "warcraft_12"]


def wmc_literal_factory(pos_weights, neg_weights, negated=False):
    """Input factory that initialises leaf logits from WMC log-weights.

    Positive literal for variable v: logits = [-inf, log_w_pos_v]
    Negated literal for variable v:  logits = [log_w_neg_v, -inf]
    """
    def factory(scope: Scope, num_units: int) -> CategoricalLayer:
        var_idx = next(iter(scope))
        if negated:
            logits = np.array([neg_weights[var_idx], -np.inf], dtype=np.float32)
        else:
            logits = np.array([-np.inf, pos_weights[var_idx]], dtype=np.float32)
        initializer = ConstantTensorInitializer(logits)
        return CategoricalLayer(
            scope,
            num_categories=2,
            num_output_units=num_units,
            logits=Parameter.from_input(TensorParameter(1, 2, initializer=initializer)),
        )
    return factory


def build_cirkit_wmc(sdd_file, ctx, weights, device):
    """Build and compile the circuit with WMC log-weights baked into the leaves.
    Returns (query, all_vars, dummy_x) ready for repeated forward passes."""
    w_pos = weights[0].squeeze().detach().cpu().numpy()
    w_neg = weights[1].squeeze().detach().cpu().numpy()

    sdd = SDD.load(sdd_file)
    sdd = sdd.build_circuit(
        literal_input_factory=wmc_literal_factory(w_pos, w_neg, negated=False),
        negated_literal_input_factory=wmc_literal_factory(w_pos, w_neg, negated=True),
    )
    sdd = ctx.compile(sdd)
    sdd = sdd.to(device).eval()

    query = IntegrateQuery(sdd)
    all_vars = Scope(list(sdd.scope))
    dummy_x = torch.zeros(1, sdd.num_variables, dtype=torch.long, device=device)
    return query, all_vars, dummy_x


def main(backend, device, nb_repeats=10, nb_warmup=2):
    ctx = PipelineContext(backend=backend, semiring='lse-sum', optimize=True, fold=True)

    for name in CIRCUITS:
        print(f"\n### Running {name} (device={device}) ###")
        sdd_file = f"experiments/nesy/circuits/{name}.sdd"

        tmp = SDD.load(sdd_file).build_circuit()
        nb_vars = tmp.num_variables
        print("number of vars", nb_vars)

        weights = _torch_weights(nb_vars, "log", device, batch_size=1)
        query, all_vars, dummy_x = build_cirkit_wmc(sdd_file, ctx, weights, device)

        fwd_timings = []
        for i in range(nb_warmup + nb_repeats):
            t0 = perf_counter()
            query(dummy_x, integrate_vars=all_vars)
            t1 = perf_counter()
            if i >= nb_warmup:
                fwd_timings.append(t1 - t0)

        fwd_bwd_timings = []
        for i in range(nb_warmup + nb_repeats):
            t0 = perf_counter()
            result = query(dummy_x, integrate_vars=all_vars).sum()
            result.backward()
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t1 = perf_counter()
            if i >= nb_warmup:
                fwd_bwd_timings.append(t1 - t0)

        fwd = np.array(fwd_timings)
        fwd_bwd = np.array(fwd_bwd_timings)
        print(f"forward:          {fwd.mean()*1e3:.3f} ± {fwd.std()*1e3:.3f} ms")
        print(f"forward+backward: {fwd_bwd.mean()*1e3:.3f} ± {fwd_bwd.std()*1e3:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default="cpu")
    parser.add_argument('-e', '--backend', type=str, default='torch', choices=["torch"])
    args = parser.parse_args()

    main(args.backend, args.device)
