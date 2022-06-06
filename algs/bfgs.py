import torch
from functools import reduce
from torch.optim.optimizer import Optimizer

from IPython import embed

# Modified from https://github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py

# Weak Wolfe conditions. Adapted from _strong_wolfe
# Does inexact line-search from https://cs.nyu.edu/overton/papers/pdffiles/bfgs_inexactLS.pdf
def _weak_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9):

    alpha = 0.0
    beta  = float('inf')

    f_eval = 1 # Counts the eval used to generate d and obj_func

    while True:
        f_eval += 1
        f_new, g_new = obj_func(x, t, d)
        gtd_new = g_new.dot(d)

        if f_new > (f + c1 * t * gtd): # If S(t) fails
            beta = t
        elif gtd_new < c2 * gtd: # If C(t) fails
            alpha = t
        else:
            break

        if beta < float('inf'):
            t = (alpha + beta) / 2.0
        else:
            t = 2.0 * alpha

    f_new, g_new = obj_func(x, t, d)
    return f_new, g_new, t, f_eval

class BFGS(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (default: 1)
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 history_size=100,
                 line_search_fn=None,
                 ls_params = None):

        defaults = dict(
            lr=lr,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super(BFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        # add line-search tolerance
        if (ls_params is not None) and 'c1' in ls_params.keys():
            self.c1 = ls_params['c1']
        else:
            self.c1 = 0

        if (ls_params is not None) and 'c2' in ls_params.keys():
            self.c2 = ls_params['c2']
        else:
            self.c2 = 0.5

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x) # Reset the change induced by _add_grad
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)

        flat_grad = self._gather_flat_grad()


        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        prev_flat_grad = state.get('prev_flat_grad')
        state['n_iter'] += 1

        ############################################################
        # compute gradient descent direction
        ############################################################
        if state['n_iter'] == 1: # this is different that original n_iter
            d = flat_grad.neg()
            old_dirs = [] # y's
            old_stps = [] # s's
            ro = []
            H_diag = 1
        else:
            if len(old_dirs) == history_size:
                # shift history by one (limited-memory)
                old_dirs.pop(0)
                old_stps.pop(0)
                ro.pop(0)

            # do lbfgs update (update memory)
            y = flat_grad.sub(prev_flat_grad)
            s = d.mul(t)
            ys = y.dot(s)  # y*s

            # store new direction/step
            old_dirs.append(y)
            old_stps.append(s)
            ro.append(1. / ys)

            if 'al' not in state: # "al"[:] = \alpha[:] on wiki
                state['al'] = [None] * history_size
            al = state['al']

            # iteration in L-BFGS loop collapsed to use just one buffer
            q = flat_grad.neg() # Notice this is *NEGATIVE* q.
            num_old = len(old_dirs)  # "m" on wiki
            for i in range(num_old - 1, -1, -1):
                al[i] = old_stps[i].dot(q) * ro[i] # alpha_i
                q.add_(old_dirs[i], alpha=-al[i])

            # update scale of initial Hessian approximation
            H_diag = ys / y.dot(y)  # (y*y), \gamma_k on wiki

            # multiply by initial Hessian
            # r/d is the final direction
            d = torch.mul(q, H_diag) # "d" = "z" on wiki

            for i in range(num_old):
                be_i = old_dirs[i].dot(d) * ro[i] #\beta_i on wiki
                d.add_(old_stps[i], alpha=al[i] - be_i)

        if prev_flat_grad is None:
            prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
        else:
            prev_flat_grad.copy_(flat_grad)
        prev_loss = loss

        ############################################################
        # compute step length
        ############################################################
        # reset initial guess for step size
        t = lr

        # directional derivative
        gtd = flat_grad.dot(d)  # g * d

        # optional line search: user function
        if line_search_fn is not None:
            # perform line search, using user function
            if line_search_fn not in ["weak_wolfe"]:
                raise RuntimeError("only 'weak_wolfe' is supported")
            else:
                ls_func = _weak_wolfe

                x_init = self._clone_param()

                def obj_func(x, t, d):
                    return self._directional_evaluate(closure, x, t, d)

                loss, flat_grad, t, self.f_eval_cur = ls_func(
                    obj_func, x_init, t, d, loss, flat_grad, gtd, c1=self.c1, c2=self.c2)
            self._add_grad(t, d)
        else:
            # no line search, simply move with fixed-step
            self._add_grad(t, d)
            self.f_eval_cur = 1

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss