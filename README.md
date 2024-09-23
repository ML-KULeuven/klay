# KLay

C++ implementation of the Knowledge Layers. Includes support for:
- SDD and d-DNNF circuits compiled by PySDD and D4.
- Evaluation in Jax and PyTorch, on CPU and GPU.
- Evaluation in various semirings (e.g. real, log, tropical).

## Installation

KLay has been tested on Linux and Mac. It requires a C++ compiler and Python 3.9 or higher.
```bash
pip install .
```

Most dependencies are optional. However, to replicate all the experiments, make sure to install the following:
```bash
pip install jax torch matplotlib numpy pysdd tqdm graphviz
```

## Experiments

The experiments in the paper can be replicated as follows. 

### Synthetic 
The synthetic experiments take a couple of hours to run.

To run all the synthetic experiments of Figure 6.
```bash
python experiments/synthetic/run_sdd.sh
```

Similarly, to run all the same synthetic experiments with d4 (as in Figure 7).
```bash
python experiments/synthetic/run_d4.sh
```
And for the synthetic experiments in the real semiring (as in Figure 8).
```bash
python experiments/synthetic/run_real.sh
```

To run the juice baseline, you need to install Julia and the juice.jl package.
Then run:
```bash
julia experiments/synthetic/benchmark_juice.jl
```

### NeSy experiments
```bash
python experiments/nesy/run.py
```

### MNIST-addition experiments
```bash
python experiments/mnist_addition/run.py -d cuda -b 128 -n 2
```
