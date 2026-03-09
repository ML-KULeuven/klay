"""Quick benchmark comparing forward/backward speed across semirings."""
import time
import torch
import klay
from pysdd.sdd import SddManager

def build_circuit(nb_vars=50):
    mgr = SddManager(var_count=nb_vars)
    variables = list(mgr.vars)
    sdd = variables[0] & variables[1]
    for v in variables[2:]:
        sdd = sdd | (sdd & v)
    c = klay.Circuit()
    c.add_sdd(sdd)
    return c, nb_vars

def bench_semiring(circuit, nb_vars, semiring, n_warmup=5, n_runs=50):
    m = circuit.to_torch_module(semiring=semiring)
    weights = torch.rand(nb_vars)

    # warmup
    for _ in range(n_warmup):
        m(weights)

    # forward
    t0 = time.perf_counter()
    for _ in range(n_runs):
        m(weights)
    fwd_time = (time.perf_counter() - t0) / n_runs

    # forward + backward
    for _ in range(n_warmup):
        m(weights.requires_grad_(True)).sum().backward()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        w = weights.detach().requires_grad_(True)
        m(w).sum().backward()
    bwd_time = (time.perf_counter() - t0) / n_runs

    return fwd_time, bwd_time

def bench_probabilistic(circuit, nb_vars, semiring, n_warmup=5, n_runs=50):
    m = circuit.to_torch_module(semiring=semiring, probabilistic=True)
    if semiring == 'log':
        weights = torch.rand(nb_vars).log()
    else:
        weights = torch.rand(nb_vars)

    for _ in range(n_warmup):
        m(weights)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        m(weights)
    fwd_time = (time.perf_counter() - t0) / n_runs

    for _ in range(n_warmup):
        m(weights.requires_grad_(True)).sum().backward()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        w = weights.detach().requires_grad_(True)
        m(w).sum().backward()
    bwd_time = (time.perf_counter() - t0) / n_runs

    return fwd_time, bwd_time

if __name__ == "__main__":
    circuit, nb_vars = build_circuit(50)
    print(f"Circuit with {nb_vars} variables\n")
    print(f"{'Semiring':<25} {'Forward (ms)':>12} {'Fwd+Bwd (ms)':>12}")
    print("-" * 52)

    for sr in ['real', 'log', 'mpe', 'godel']:
        fwd, bwd = bench_semiring(circuit, nb_vars, sr)
        print(f"{sr:<25} {fwd*1000:>12.3f} {bwd*1000:>12.3f}")

    for sr in ['real', 'log']:
        fwd, bwd = bench_probabilistic(circuit, nb_vars, sr)
        print(f"prob-{sr:<20} {fwd*1000:>12.3f} {bwd*1000:>12.3f}")
