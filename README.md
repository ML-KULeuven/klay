# KLay

C++ implementation of the Knowledge Layers. Includes support for:
- SDD and d-DNNF circuits compiled by PySDD and D4.
- Evaluation in Jax and PyTorch, on CPU and GPU.
- Evaluation in various semirings (e.g. real, log, tropical).

## Installation

KLay has been tested on Linux (x86) and Mac (x86/ARM), no guarantees are made for Windows. KLay requires a C++ compiler and Python 3.9 or higher.
```bash
pip install .
```

Most dependencies are optional. However, to replicate all the experiments, make sure to install the following:
```bash
pip install jax torch torchvision matplotlib numpy pysdd tqdm graphviz
```

## Experiments

The experiments in the paper can be replicated as follows. 

### Synthetic 
The synthetic experiments take a couple of hours to run.

To run all the synthetic experiments of Figure 6.
```bash
bash experiments/synthetic/run_sdd.sh
```

Similarly, to run all the same synthetic experiments with d4 (as in Figure 7).
```bash
bash experiments/synthetic/run_d4.sh
```
And for the synthetic experiments in the real semiring (as in Figure 8).
```bash
bash experiments/synthetic/run_real.sh
```

To run the juice baseline, install [Julia](https://julialang.org/) and the [LogicCircuits.jl](https://github.com/Tractables/LogicCircuits.jl) package.
Then run:
```bash
julia experiments/synthetic/benchmark_juice.jl
```

You can reproduce the plots of the paper using:
```bash
python experiments/synthetic/plot_figure.py
```

### NeSy experiments
```bash
python experiments/nesy/run.py
```

### MNIST-addition experiments
```bash
python experiments/mnist_addition/run.py -d cuda -b 128 -n 2
```
