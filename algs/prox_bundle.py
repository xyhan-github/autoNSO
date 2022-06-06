# Proximal Bundle Algorithm

import numpy as np
import cvxpy as cp

from algs.optAlg import OptAlg

from IPython import embed

# Proximal Bundle Algorithm
class ProxBundle(OptAlg):
    def __init__(self, objective, mu=1.0, null_k=0.5, ignore_null=False, prune='Drop Inactive', active_thres=5e-3,
                 S = None, **kwargs):

        super(ProxBundle, self).__init__(objective, **kwargs)
        assert prune in ['Full','Drop Inactive','Drop All']

        self.constraints    = []
        self.p              = cp.Variable(self.x_dim)  # variable of optimization
        self.v              = cp.Variable()  # value of cutting plane model
        self.mu             = mu
        self.null_k         = null_k
        self.prune          = prune
        self.active_thres   = active_thres
        self.objective.oracle_output = 'both'
        self.serious        = False

        self.update_name()

        self.cur_x          = self.x0
        self.cur_y          = self.x0  # the auxiliary variables will null values
        self.path_y         = np.array([],dtype=np.float64).reshape(0,self.x_dim)
        self.ignore_null    = ignore_null # Whether to save null steps in path
        self.cur_active     = np.array([0])

        # Create the bundle
        self.make_bundle(S)

        self.update_params(None)

    def make_bundle(self, S):
        if S is None:
            S = self.x0[np.newaxis,:]

        for i in range(S.shape[0]): # Add bundle
            orcl_call = self.objective.call_oracle(S[i,:])
            self.constraints += [orcl_call['f'].copy() + orcl_call['df'].copy() @ (self.p - S[i,:].copy()) <= self.v]


    def step(self):

        super(ProxBundle, self).step()

        prox_objective = self.v + (self.mu / 2.0) * cp.quad_form(self.p - self.cur_x, np.eye(self.x_dim))
        prob = cp.Problem(cp.Minimize(prox_objective), self.constraints)
        self.p.value = self.cur_y  # Warm-starting
        prob.solve()

        # Update current iterate value and update the bundle
        self.cur_y = self.p.value

        # Find number of tight constraints
        duals = [self.constraints[i].dual_value for i in range(len(self.constraints))]
        thres = self.active_thres * max(duals)

        self.cur_active = np.where([(duals[i] > thres) for i in range(len(self.constraints))])[0]

        # if len(self.cur_active) == 3:
        #     embed()

        # Update paths and bundle constraints
        self.update_params(self.v.value)

    def update_params(self, expected):

        self.path_y = np.vstack([self.path_y, self.cur_y])
        orcl_call = self.objective.call_oracle(self.cur_y)
        cur_fy = orcl_call['f']

        # Whether to take a serious step
        if expected is not None:
            self.serious = ((self.path_fx[-1] - cur_fy) > self.null_k * (self.path_fx[-1] - expected))
        else:
            self.serious = True

        if self.serious: # If changing basis. Requires at least a serious step
            self.cur_x = self.cur_y.copy()

            if self.ignore_null: # Do not save null in path_x, just save copies of x
                self.cur_fx = orcl_call['f'].copy()
                self.path_x = np.vstack([self.path_x, self.cur_x])

        if not self.ignore_null: # Save null in the path_x
            self.path_x = self.path_y

        self.update_fx_step()
        self.path_fx = np.vstack([self.path_fx, self.cur_fx])

        if expected is not None:
            self.cur_iter += 1 # Count null steps as iterations

        super(ProxBundle, self).update_params() # Check fx_step size save bundle if necessary

        if self.prune == 'Drop Inactive': # Remove inactive constraints
            inactive = np.setdiff1d(np.arange(len(self.constraints)),self.cur_active)[::-1] # Removes in descending order
            [self.constraints.pop(i) for i in inactive]
        elif self.prune == 'Drop All':
            if self.serious:
                self.constraints = []
        # Even if it is null step, add a constraint to cutting plane model
        self.constraints += [(cur_fy.copy() + orcl_call['df'].copy() @ (self.p - self.cur_y.copy())) <= self.v]

    def update_fx_step(self):
        old_fx = self.cur_fx.copy() if (self.cur_fx is not None) else float('inf')
        self.cur_fx = self.objective.obj_func(self.cur_x).data.numpy()
        self.fx_step = (old_fx - self.cur_fx)

    def update_name(self):
        print("Prune Bundle: {}".format(self.prune), flush=True)
        self.name = 'ProxBundle'
        self.name += (' [' + self.prune + ']')
        self.name += ' (null-k=' + str(self.null_k) + ')'
        self.name += r' ($\mu={}$)'.format(self.mu)