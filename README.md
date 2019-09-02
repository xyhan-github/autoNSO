# autoNSO
Naive implementations of common non-smooth optimization (NSO) methods using PyTorch auto-differentiation as the subgradient oracle.  

Many academic works assume one does not "see" the explicit form of the objective function of an optimization problem. Instead, one can only evaluate the function at certain points as well as obtain one subgradient in the subdifferential of the function at that point ("make oracle calls"). Under that setting, autoNSO is designed for comparing and benchmarking the convergence rates and step-wise optimization paths of common NSO algorithms on an arbitrary objective function.  

Maintainer:   X.Y. Han, Cornell University ORIE\
Contact:      xiaoyanhn@gmail.com

Please cite:

> X.Y. Han, *autoNSO: Naive Implementations of Common NSO Methods with Auto-differentiation*, (2019), GitHub repository, https://github.com/xiaoyanh/autoNSO

```
@misc{Han2019,
  author = {Han, X.Y.},
  title = {autoNSO: Naive Implementations of Common NSO Methods with Auto-differentiation},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xiaoyanh/autoNSO}},
  commit = {<ADD COMMIT ID HERE>}
}
```

Disclaimer: This code is essentially a wrapper between PyTorch, CVXPY, and Matplotlib to do benchmarking for NSO research. It is not implemented to be the fastest possible in terms of wall-clock time.
