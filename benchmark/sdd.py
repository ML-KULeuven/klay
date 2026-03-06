from array import array
from time import perf_counter

from .utils import python_weights


def benchmark_pysdd(sdd, nb_vars, semiring, nb_repeats=10, device='cpu'):
    assert device == 'cpu'
    pos_weights, neg_weights = python_weights(nb_vars, semiring)
    # WARNING: pysdd computes both the forward and backward passes in propagate
    pysdd_weights = array('d', neg_weights[::-1] + pos_weights)
    wmc_manager = sdd.wmc(log_mode=(semiring == "log"))
    wmc_manager.set_literal_weights_from_array(pysdd_weights)

    timings = []
    for _ in range(nb_repeats+2):
        t1 = perf_counter()
        wmc_manager.propagate()
        timings.append(perf_counter() - t1)
    return {'backward': timings[2:]}
