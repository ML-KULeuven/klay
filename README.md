# KLay

[![Python 3.11](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/klaycircuits?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/klaycircuits)

__KLay is a Python library for evaluating sparse circuits on the GPU.__

To get started, install KLay using pip and check out the [documentation](https://klaycircuits.readthedocs.io/en/latest/). You can also refer to [this video](https://www.youtube.com/watch?v=ZCpDenbGoJ4) or [the paper](https://openreview.net/pdf?id=Zes7Wyif8G) for more information.

```bash
pip install klaycircuits
```

KLay features include:
- GPU acceleration of circuits using Jax or PyTorch. 
- Loading SDD and d-DNNF circuits compiled by [PySDD](https://github.com/ML-KULeuven/PySDD) or [D4](https://github.com/crillab/d4).
- Evaluation in various semirings (e.g. real, log, tropical).
- Propagating constants and merging duplicate nodes.

The JAX evaluator uses native JAX segment reductions by default. NVIDIA CUDA
users can select the optional `SpringMV` sparse backend for the `log` and
`godel` semirings:

```python
func = circuit.to_jax_function(semiring="log", backend="springmv")
```

SpringMV is currently built from source. Install CUDA-enabled JAX first, then
install SpringMV from its public repository:

```bash
pip install --upgrade "jax[cuda13]"
pip install "springmv @ git+https://github.com/pedrozudo/springmv.git"
```

The source build requires CMake 3.24+, a C++17 compiler, and a CUDA toolkit
containing `nvcc`. If CUDA is not in its default location, pass it explicitly:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCUDAToolkit_ROOT=/usr/local/cuda" \
pip install "springmv @ git+https://github.com/pedrozudo/springmv.git"
```


## 🧪 Tests

Run the test suite from the project root:

```bash
pytest tests/
```

Tests are split by backend. They are automatically skipped if the required backend is not installed:
- `tests/test_manual.py`, `tests/fuzzer_torch.py`, `tests/fuzzer_torch_multi.py`, `tests/fuzzer_creation.py` — require PyTorch
- `tests/fuzzer_jax.py` — requires JAX
- `tests/test_compression.py` — no backend required

## 📊 Benchmarks

Benchmarks live in the `benchmark/` directory. Run from the project root with:

```bash
python -m benchmark.benchmark_wmc --benchmark sdd --target torch -v 100 200 500
python -m benchmark.benchmark_wmc --benchmark sdd --target jax   -v 100 200 500
python -m benchmark.benchmark_wmc --benchmark sdd --target pysdd -v 100 200 500
```

Compare native JAX segment reductions with SpringMV on the same circuit and
resident GPU inputs:

```bash
python -m benchmark.benchmark_wmc \
  --benchmark sdd --target jax --jax-backend both \
  --semiring log --device cuda:0 --nb_vars 100 \
  --batch_size 128 --warmup 10 --iterations 50
```

Key options:

| Flag | Description |
|------|-------------|
| `-b` / `--benchmark` | Circuit type: `sdd` or `d4` |
| `-t` / `--target` | Backend: `torch`, `jax`, or `pysdd` |
| `-v` / `--nb_vars` | Number of variables (one or more) |
| `-d` / `--device` | Device: `cpu`, `cuda`, `cuda:0`, etc. |
| `-s` / `--semiring` | Semiring: `log` (default) or `real` |
| `-r` / `--nb_repeats` | Number of seeds to average over (default: 1) |
| `--jax-backend` | JAX sparse backend: `jax`, `springmv`, or `both` |
| `--warmup` | Untimed warm-up executions per JAX backend |
| `--iterations` | Timed executions per JAX backend |

Results are saved as JSON files under `results/`.

## 📃 Paper

If you use KLay in your research, consider citing [our paper](https://openreview.net/pdf?id=Zes7Wyif8G).

To replicate the exact results and figures of the paper, use [this code](https://github.com/ML-KULeuven/klay/tree/d3b81491c34603ba9271d25af7c789d3ba368ede).

```bibtex
@inproceedings{
    maene2025klay,
    title={{KL}ay: Accelerating Arithmetic Circuits for Neurosymbolic {AI}},
    author = {Maene, Jaron and Derkinderen, Vincent and Zuidberg Dos Martires, Pedro},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=Zes7Wyif8G}
}
```

## 🚀 Impact

Examples of papers that use KLay include:

- [Auto-Formalizing Neuro-Symbolic Predictors](https://arxiv.org/pdf/2610.01519)
- [Neurosymbolic Reinforcement Learning: Playing MiniHack with Probabilistic Logic Shields](https://ojs.aaai.org/index.php/AAAI/article/view/35349)
- [ExplainFuzz: Explainable and Constraint-Conditioned Test Generation with Probabilistic Circuits](https://arxiv.org/abs/2604.06559)
- [ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](https://ojs.aaai.org/index.php/AAAI/article/view/39033)
- [Neurosymbolic Object-Centric Learning with Distant Supervision](https://arxiv.org/abs/2506.16129)
- [The DeepLog Neurosymbolic Machine](https://arxiv.org/abs/2508.13697)
- [Rule-Selection Surrogates for Tractable Neurosymbolic Inference](https://openreview.net/pdf?id=12M7z6mhOV)
- [Tractable Probabilistic Neurosymbolic AI Using Boolean Tensor Factorizations](https://openreview.net/pdf?id=FflbKkBH0z)
