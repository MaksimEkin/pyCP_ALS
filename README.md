# pyCP_ALS

<div align="center", style="font-size: 50px">
    <img src="https://github.com/MaksimEkin/pyCP_ALS/actions/workflows/ci_tests.yml/badge.svg?branch=main"></img>
    <img src="https://img.shields.io/hexpm/l/plug"></img>
    <img src="https://img.shields.io/badge/python-v3.8.5-blue"></img>
</div>

<br>


pyCP_ALS is the Python implementation of **CP-ALS** algorithm that was originally introduced in the [MATLAB Tensor Toolbox](https://www.tensortoolbox.org/cp.html>) [1,2]. 

<div align="center", style="font-size: 50px">

### [:orange_book: Example Notebooks](examples/) 

</div>


## Installation

#### Option 1: Install using *pip*
```shell
pip install git+https://github.com/MaksimEkin/pyCP_ALS.git
```

#### Option 2: Install from source
```shell
git clone https://github.com/MaksimEkin/pyCP_ALS.git
cd pyCP_ALS
conda create --name pyCP_ALS python=3.8.5
source activate pyCP_ALS
python setup.py install
```

## Example Usage
```python
from pyCP_ALS import CP_ALS
import pickle

data = pickle.load(open("data/toy_tensor.p", "rb"))
nnz_values = data["nnz_values"]
nnz_coords = data["nnz_coords"]

model = CP_ALS(
    tol=1e-4, 
    n_iters=50, 
    verbose=True, 
    fixsigns=True, 
    random_state=42,
)

M = model.fit(
    coords=nnz_coords, 
    values=nnz_values, 
    rank=2, 
    Minit="random",
)
```
**See the [examples](examples/) for more.**

## How to Cite pyCP_ALS?
If you use pyCP_ALS, please cite it.

```latex
@MISC{Eren2022pyCP_ALS,
  author = {M. E. {Eren}},
  title = {{pyCP_ALS}},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MaksimEkin/pyCP_ALS}}
}
```

## Developer Test Suite
Developer test suites are located under [```tests/```](tests/) directory. Tests can be ran from this folder using ```python -m unittest *```.

## References
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.

[2] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.


