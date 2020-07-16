import numpy as np

def hess_approx_cI(hess):
    sigmas = np.linalg.svd(hess, compute_uv=False)
    mu = np.mean(sigmas[sigmas > max(sigmas) * 1e-6])
    return mu * np.eye(hess.shape[0])