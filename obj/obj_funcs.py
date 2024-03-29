import torch
import numpy as np
import cvxpy as cp
from IPython import embed
from obj.objective import Objective
from torch import abs, max, sum, norm, sqrt, einsum, stack, symeig, tensor, Tensor

def simple2D(x):
    return max(abs(x[0]),(0.5 * x[1]**2))
Simple2D = Objective(simple2D)

def partlysmooth2D(x):
    if type(x) != Tensor:  # If non-tensor passed in, no gradient will be used
        x = tensor(x, dtype=torch.double, requires_grad=False)
    assert len(x) == 2
    return max(3*x[0]**2+x[1]**2-x[1],x[0]**2+x[1]**2+x[1])
PartlySmooth2D = Objective(partlysmooth2D)

def partlysmooth3D(x):
    if type(x) != Tensor:  # If non-tensor passed in, no gradient will be used
        x = tensor(x, dtype=torch.double, requires_grad=False)
    assert len(x) == 3
    return sqrt((x[0]**2 - x[1])**2 + x[2]**2) + 2*(x[0]**2 + x[1]**2 + x[2]**2)
PartlySmooth3D = Objective(partlysmooth3D)

def convex3D(x):
    if type(x) != Tensor:  # If non-tensor passed in, no gradient will be used
        x = tensor(x, dtype=torch.double, requires_grad=False)
    assert len(x) == 3
    return sqrt((x[0]**2 - x[1])**2 + x[2]**2) + x[0]**2
Convex3D = Objective(convex3D)


# Below are example objective functions from Lewis-Wylie 2019 (https://arxiv.org/abs/1907.11742)

# Creates a strongly convex objective function for particular n and k
def stronglyconvex(n=50, k=10, seed=0, **kwargs):
    torch.random.manual_seed(seed)

    # Generate the g's, so that they sum with positive weights to 0
    lam = torch.rand(k,dtype=torch.double)
    lam /= sum(lam)
    g  = torch.randn(k - 1, n, dtype=torch.double)
    gk = -(lam[0:(k-1)] @ g)/lam[-1]
    g  = torch.cat((g, gk[None, :]), 0)

    c = torch.randn(k,dtype=torch.double)
    tmp = torch.randn(k,n,n,dtype=torch.double)
    H = stack([tmp[i,:,:].T @ tmp[i,:,:] for i in range(k)])

    def sc_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        term1 = g@x
        term2 = 0.5 * stack([x.T @ H[i,:,:] @ x for i in range(k)])
        term3 = (1./24.) * (norm(x)**4) * c
        return max(term1+term2+term3)

    return Objective(sc_function, **kwargs)

# Creates a nonconvex objective function for particular n and k
def nonconvex(n=50, k=10, seed=0, **kwargs):
    torch.random.manual_seed(seed)

    lam = torch.rand(k,dtype=torch.double)
    lam /= sum(lam)
    g  = torch.randn(k - 1, n, dtype=torch.double)
    gk = -(lam[0:(k-1)] @ g)/lam[-1]
    g  = torch.cat((g, gk[None, :]), 0)

    c = torch.randn(k,dtype=torch.double)
    tmp = torch.randn(k,n,n,dtype=torch.double)
    H = stack([tmp[i,:,:].T @ tmp[i,:,:] for i in range(k)])

    def nc_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        term1 = g@x
        term2 = 0.5 * stack([x.T @ H[i,:,:] @ x for i in range(k)])
        term3 = (1./24.) * (norm(x)**4) * c
        return sum(abs(term1 + term2 + term3)[0:k])

    return Objective(nc_function, **kwargs)

# Creates a partly objective function for particular n and m
def partlysmooth(n=50, m=25, seed=0, **kwargs):
    torch.random.manual_seed(seed)
    tmp = torch.randn(n+1,m,m,dtype=torch.double)
    A = stack([tmp[i, :, :].T + tmp[i, :, :] for i in range(n+1)])

    # Get true vaues
    l = cp.Variable(n)
    obj = A[0,:,:].data.numpy()
    for i in range(n):
        obj += A[i+1,:,:]*l[i]
    prob = cp.Problem(cp.Minimize(cp.lambda_max(obj)))
    prob.solve(solver='MOSEK')

    true_val = prob.value
    true_spec = np.linalg.eigvalsh(A[0,:,:] + np.einsum('i,ijk->jk',l.value,A[1:,:,:]))
    true_mult = np.sum(np.isclose(true_spec,np.max(true_spec)))

    def ps_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        mat = A[0,:,:] + einsum('i,ijk->jk',x,A[1:,:,:])
        return symeig(mat,eigenvectors=True)[0][-1] # eigenvalues in ascending order

    return Objective(ps_function, **kwargs), true_val, true_mult

# Half and half
def halfandhalf(n=50, seed=0, **kwargs):
    A = torch.ones(n,dtype=torch.double)
    A[1::2] = 0
    A = torch.diag(A)
    B = torch.diag((torch.arange(n,dtype=torch.double)+1.0)**-1)

    def hh_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        return sqrt(x.T@A@x) + x.T@B@x

    return Objective(hh_function, **kwargs)

# Creates a strongly convex objective function for particular n and k
def maxQuadratic(n=50, k=10, seed=0, L = 10, extra=False, isotropic=False, min_cond=0, **kwargs):
    torch.random.manual_seed(seed)

    # # Generate the g's, so that they sum with positive weights to 0
    lam = (torch.rand(k,dtype=torch.double)*(1-min_cond)) + min_cond
    lam /= sum(lam)
    g  = torch.randn(k - 1, n, dtype=torch.double)
    gk = -(lam[0:(k-1)] @ g)/lam[-1]
    g  = torch.cat((g, gk[None, :]), 0)

    if isotropic:
        H = stack([torch.eye(n) * torch.rand(1) * L/(i+1) for i in range(k)])
    else:
        tmp = torch.randn(k, n, n, dtype=torch.double)
        tmp = stack([tmp[i,:,:].T @ tmp[i,:,:] for i in range(k)])
        max_norm = np.max([float(np.linalg.norm(tmp[i,:,:],ord=2)) for i in range(k)])
        H = stack([(L / max_norm) * tmp[i,:,:] for i in range(k)])

    if extra:
        L_vec = [float(torch.svd(H[i,:,:],compute_uv=False)[1][0]) for i in range(k)]

    def quad_function(x):
        if type(x) != Tensor: # If non-tensor passed in, no gradient will be used
            x = tensor(x, dtype=torch.double, requires_grad=False)
        assert len(x) == n

        term1 = g@x
        # term1 = 0
        term2 = 0.5 * stack([x.T @ H[i,:,:] @ x for i in range(k)])
        return max(term1+term2)

    attr = {'lambda' : lam,
            'g'      : g}

    if extra:
        extra_info = {'L': L_vec,
                      'lambda': lam,
                      'H' : H,
                      'g' : g}
        obj = Objective(quad_function, **kwargs)
        obj.extra_info = extra_info
        return obj
    else:
        return Objective(quad_function, **kwargs)

