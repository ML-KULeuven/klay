# KLay

[![Python 3.10](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build](https://github.com/ML-KULeuven/klay/actions/workflows/main.yml/badge.svg)](https://github.com/ML-KULeuven/klay/actions/workflows/main.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/klaycircuits?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/klaycircuits)

__KLay is a Python library for evaluating sparse circuits on the GPU.__

To get started, install KLay using pip and check out the [documentation](https://klaycircuits.readthedocs.io/en/latest/). You can also refer to [this video](https://www.youtube.com/watch?v=ZCpDenbGoJ4) or [the paper](https://openreview.net/pdf?id=Zes7Wyif8G) for more information.

```bash
pip install klaycircuits
```

KLay features:
- GPU acceleration of circuits using Jax or PyTorch. 
- Loading SDD and d-DNNF circuits compiled by [PySDD](https://github.com/ML-KULeuven/PySDD) or [D4](https://github.com/crillab/d4).
- Evaluation in various semirings (e.g. real, log, tropical).
- Propagating constants and merging duplicate nodes.


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

Key options:

| Flag | Description |
|------|-------------|
| `-b` / `--benchmark` | Circuit type: `sdd` or `d4` |
| `-t` / `--target` | Backend: `torch`, `jax`, or `pysdd` |
| `-v` / `--nb_vars` | Number of variables (one or more) |
| `-d` / `--device` | Device: `cpu`, `cuda`, `cuda:0`, etc. |
| `-s` / `--semiring` | Semiring: `log` (default) or `real` |
| `-r` / `--nb_repeats` | Number of seeds to average over (default: 1) |

Results are saved as JSON files under `results/`.

### Regression benchmarks

Per-commit regression benchmarks use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/). Run from the project root:

```bash
uv run pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=<name>
```

Compare two saved runs:

```bash
uv run pytest-benchmark compare <old> <new> --columns=mean,stddev
```

Results are saved as JSON files under `.benchmarks/`.

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
