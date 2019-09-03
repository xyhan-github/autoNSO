# autoNSO
Naive implementations of common non-smooth optimization (NSO) methods using PyTorch auto-differentiation as the subgradient oracle.  

Many academic works assume one does not "see" the explicit form of the objective function of an optimization problem. Instead, one can only evaluate the function at certain points as well as obtain one subgradient in the subdifferential of the function at that point ("make oracle calls"). Under that setting, autoNSO is designed for comparing and benchmarking the convergence rates and step-wise optimization paths of common NSO algorithms on an arbitrary objective function. This code is unlike most NSO code in that one **does not need to define the subgradient oracle**; instead, the subgradient is calculated automatically using PyTorch autodifferentiation.

Currently, the software is capable of the following methods:

* Prox-Bundle
* Subgradient Descent
* Nesterov Accelerated Subgradient Descent
* L-BFGS

Defining new optimization algorithms for one's own use is easy: just modify the code in `algs/optAlg.py`.

Quick examples are in the  `simple_examples`  folder. In particular, the following two plots were generated using only 14 lines of code (see `simple_examples/plot_multiple.py`) for the objective function $f(x,y) = \max(|x|,y^2)$:

![](https://raw.githubusercontent.com/xiaoyanh/autoNSO/master/aux/path_plot_ex.png= 250x250) 

Maintainer:   X.Y. Han, Cornell University ORIE\
Contact:      xiaoyanhn@gmail.com

Please cite:

> X.Y. Han, *autoNSO: Implementations of Common NSO Methods with Auto-differentiation*, (2019), GitHub repository, https://github.com/xiaoyanh/autoNSO

```
@misc{Han2019,
  author = {Han, X.Y.},
  title = {autoNSO: Implementations of Common NSO Methods with Auto-differentiation},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xiaoyanh/autoNSO}},
  commit = {<ADD COMMIT ID HERE>}
}
```

Disclaimer: This code is essentially a wrapper between PyTorch, CVXPY, and Matplotlib to do benchmarking for NSO research. It is not implemented to be the fastest possible in terms of wall-clock time.
