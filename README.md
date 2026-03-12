# ASEAMS Package

**ASEAMS** is a Python toolkit implementing **Adaptive Multilevel Splitting (AMS)** techniques for molecular dynamics simulations, including:
- Collective variables (`cvs`)
- Initial condition samplers (`inicondsamplers`)
- AMS routine (`ams`)
- Langevin integrators (`utils`)

It is designed to work seamlessly with the [ASE](https://wiki.fysik.dtu.dk/ase/) molecular simulation framework.

---

## 🚀 Installation

Install directly from GitHub:

```bash
pip install "aseams @ git+https://github.com/thomaspigeon/aseams.git"
```

Install from a local checkout:

```bash
git clone https://github.com/thomaspigeon/aseams.git
cd aseams
pip install .
```

Install in editable mode for development:

```bash
pip install -e .[dev]
```

Use it from Python with the standard installed import path:

```python
from aseams import AMS, CollectiveVariables, SingleWalkerSampler
```

Example dependency declarations:

`requirements.txt`

```text
aseams @ git+https://github.com/thomaspigeon/aseams.git
```

`pyproject.toml`

```toml
dependencies = [
    "aseams @ git+https://github.com/thomaspigeon/aseams.git",
]
```

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
