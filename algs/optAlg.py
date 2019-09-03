#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""
import numpy as np
import cvxpy as cp
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from IPython import embed

class OptAlg:
    def __init__(self, objective, max_iter = 1000, x0 = None, verbose=True):
        
        if type(x0) is not np.ndarray:
            x0 = np.array(x0)
        
        assert len(x0.shape) == 1 # assert initial is vector
        
        self.max_iter   = max_iter
        self.objective  = objective
        
        self.x0         = x0
        self.x_dim      = self.x0.shape[0]
        
        self.cur_iter  = 0
        self.cur_x     = None
        self.cur_fx    = None
        self.path_x      = None
        self.path_fx     = None
        
        self.verbose = verbose
        
        self.opt_x     = None
        self.opt_fx    = None
        self.total_iter = None
        
        self.name = None
    
    def optimize(self):

        # Run the optimization algorithm until a stopping condition is hit
        stop = False
        for i in range(self.max_iter):
            self.step()
            stop = self.stop_cond()
            
            if stop:
                break

        self.opt_x = self.path_x[-1]
        self.opt_fx = self.path_fx[-1]
        self.total_iter = self.cur_iter
        
        if self.verbose:
            print('Optimal Value: ' + str(self.opt_fx))
            print('Optimal Point: ' + str(self.opt_x))
    
    def step(self):
        if self.verbose:
            print('iter: ' + str(self.cur_iter))
    
    def stop_cond(self):
        return False
    
    def update_params(self):
        pass
    
        
class ProxBundle(OptAlg):
    def __init__(self, objective, max_iter = 10, x0 = None, mu = 1.0):
        super(ProxBundle,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.constraints    = []
        self.p              = cp.Variable(self.x_dim) # variable of optimization
        self.v              = cp.Variable() # value of cutting plane model
        self.mu             = mu
        self.name           = 'ProxBundle'
        self.name += ' (mu='+str(self.mu)+')'
        
        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.update_params()

    def step(self):
        
        super(ProxBundle,self).step()
        
        prox_objective = self.v + 0.5 * (1.0/(2.0 * self.mu)) * cp.power(cp.norm(self.p - self.cur_x,2),2)
        
        prob = cp.Problem(cp.Minimize(prox_objective),self.constraints)
        prob.solve()
        
        # Update current iterate value and update the bundle
        self.cur_x      = self.p.value
        self.update_params()
        
        # Update paths and bundle constraints
        self.cur_iter    += 1
    
    def update_params(self):
        
        orcl_call           = self.objective.call_oracle(self.cur_x)
        self.cur_fx         = orcl_call['f']
        self.constraints    += [(self.cur_fx.copy() + 
                                 orcl_call['df'].copy()@(self.p - self.cur_x.copy())) <= self.v]
    
        if self.path_x is not None:
            self.path_x         = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx        = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
        else:
            self.path_x         = self.cur_x[np.newaxis]
            self.path_fx        = self.cur_fx[np.newaxis]
        
# Subgradient method
class TorchAlg(OptAlg):
    def __init__(self, objective, max_iter=10, x0 = None):
        super(TorchAlg,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        # Add one bundle point to initial point
        self.cur_x  = self.x0
        self.update_params()
        
        # Set up criterion and thing to be optimized
        self.criterion = self.objective.obj_func
        self.p         = torch.tensor(self.x0,dtype=torch.float,requires_grad=True)
        
    def step(self):
        
        super(TorchAlg,self).step()
        
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        value = self.criterion(self.p)
        value.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Update current iterate value and update the bundle
        self.cur_x      = self.p.data.numpy().copy()
        self.update_params()
        
        # Update paths and bundle constraints
        self.cur_iter    += 1
    
    def update_params(self):
        
        self.cur_fx = self.objective.call_oracle(self.cur_x)['f']
    
        if self.path_x is not None:
            self.path_x         = np.concatenate((self.path_x, self.cur_x[np.newaxis]))
            self.path_fx        = np.concatenate((self.path_fx, self.cur_fx[np.newaxis]))
        else:
            self.path_x         = self.cur_x[np.newaxis]
            self.path_fx        = self.cur_fx[np.newaxis]
            
class Subgradient(TorchAlg):
    def __init__(self, objective, max_iter=10, x0 = None, lr = 1, decay=0.9):
        super(Subgradient,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.lr = lr
        self.decay = decay
        self.name = 'Subgradient'
        self.name += ' (lr=' + str(self.lr)+',decay='+str(self.decay)+')'
        
        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=0)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)

class Nesterov(TorchAlg):
    def __init__(self, objective, max_iter=10, x0 = None, lr = 1, decay=0.9, momentum=0.9):
        super(Nesterov,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.lr = lr
        self.decay = decay
        self.momentum = 0.9
        self.name = 'Nesterov'
        self.name += (' (lr=' + str(self.lr)+',decay='+str(self.decay)
                        +',mom='+str(self.momentum)+')')
        
        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.SGD([self.p], lr=self.lr, momentum=self.momentum)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)
        
class LBFGS(TorchAlg):
    def __init__(self, objective, max_iter=10, x0 = None, lr = 1, decay=0.9, hist=100):
        super(LBFGS,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.lr = lr
        self.decay = decay
        self.hist  = hist
        self.name = 'LBFGS'
        self.name += (' (lr=' + str(self.lr)+',decay='+str(self.decay)
                        +',hist='+str(self.hist)+')')
        
        # SGD without batches and momentum reduces to subgradient descent
        self.optimizer = optim.LBFGS([self.p], lr=self.lr, history_size=self.hist)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.decay)
        
    def step(self):
        
        super(TorchAlg,self).step()
        
        def closure():
            self.optimizer.zero_grad()
            value = self.criterion(self.p)
            value.backward()
            return value
        
        self.optimizer.step(closure)
        self.scheduler.step()
        
        # Update current iterate value and update the bundle
        self.cur_x      = self.p.data.numpy().copy()
        self.update_params()
        
        # Update paths and bundle constraints
        self.cur_iter    += 1
        
    