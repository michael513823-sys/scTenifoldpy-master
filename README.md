# scTenifoldpy
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/biodbs.svg)](https://pypi.python.org/pypi/biodbs/)
[![Pattern](https://img.shields.io/badge/DOI-10.1016/j.patter.2020.100139-blue)](https://www.sciencedirect.com/science/article/pii/S2666389920301872#bib48)
[![GitHub license](https://img.shields.io/github/license/qwerty239qwe/scTenifoldpy.svg)](https://github.com/qwerty239qwe/scTenifoldpy/blob/master/LICENSE)

> **This is a fork of [qwerty239qwe/scTenifoldpy](https://github.com/qwerty239qwe/scTenifoldpy).**
> Key improvements over the original:
> - **Python 3.9+ compatible** — works with modern Python environments
> - **Updated dependencies** — compatible with the latest pandas, NumPy, and scanpy
> - **CUDA GPU acceleration** — optional GPU backend via CuPy or PyTorch for PCNet construction, tensor decomposition, and virtual knock-out propagation

This package is a Python version of [scTenifoldNet](https://github.com/cailab-tamu/scTenifoldNet) 
and [scTenifoldKnk](https://github.com/cailab-tamu/scTenifoldKnk). If you are a R/MATLAB user, 
please install them to use their functions. 
Also, please [cite](https://www.sciencedirect.com/science/article/pii/S2666389920301872) the original paper properly 
if you are using this in a scientific publication. Thank you!

---

### Installation

**Clone and install locally:**
```bash
git clone https://github.com/michael513823-sys/scTenifoldpy-cuda.git
cd scTenifoldpy
pip install -e .
```

---

### Usages

scTenifold can be imported as a normal Python package:

#### scTenifoldNet
```python
from scTenifold.data import get_test_df
from scTenifold import scTenifoldNet

df_1, df_2 = get_test_df(n_cells=1000), get_test_df(n_cells=1000)
sc = scTenifoldNet(df_1, df_2, "X", "Y", qc_kws={"min_lib_size": 10})
result = sc.build()
```

#### scTenifoldKnk
```python
from scTenifold.data import get_test_df
from scTenifold import scTenifoldKnk

df = get_test_df(n_cells=1000)
sc = scTenifoldKnk(data=df,
                   ko_method="default",
                   ko_genes=["NG-1"],  # the gene you want to knock out
                   qc_kws={"min_lib_size": 10, "min_percent": 0.001},
                   )
result = sc.build()
```

---

### CUDA GPU Acceleration

This fork adds optional CUDA acceleration for the three most compute-intensive steps. Enable it by passing `use_cuda=True`:

#### PCNet construction (network building)
```python
# Uses GPU batch-SVD instead of Ray multiprocessing
sc.run_step("nc", use_cuda=True)

# Specify a particular GPU
sc.run_step("nc", use_cuda=True, device="cuda:1")
```

#### Tensor decomposition
```python
sc.run_step("td", use_cuda=True)
```

#### Virtual knock-out with propagation
```python
sc = scTenifoldKnk(
    data=df,
    ko_method="propagation",
    ko_genes=["NG-1"],
    nc_kws={"use_cuda": True},
    td_kws={"use_cuda": True},
    ko_kws={"use_cuda": True},
)
result = sc.build()
```

**Automatic fallback:** if no CUDA device is detected at runtime, a `RuntimeWarning` is printed and computation falls back to the original CPU path automatically.

| Parameter | Default | Description |
|---|---|---|
| `use_cuda` | `False` | Enable GPU acceleration |
| `device` | `"cuda:0"` | CUDA device string (PyTorch backend only) |
| `fast_svd` | `True` | Use batched global-SVD approximation (fastest, CuPy only) |

See [parallel_computing_analysis.md](parallel_computing_analysis.md) for a detailed technical analysis of the original Ray-based parallelism and the CUDA redesign.

---

### Command Line tool
Once the package is installed, users can use the command line tool to generate all results.

Use this command to create a config file:
```shell
python -m scTenifold config -t 1 -p ./net_config.yml
```
Open the config file, add data path, and edit the parameters. Then run:

```shell
# scTenifoldNet
python -m scTenifold net -c ./net_config.yml -o ./output_folder

# scTenifoldKnk
python -m scTenifold knk -c ./knk_config.yml -o ./output_folder
```
