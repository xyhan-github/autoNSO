#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""
import numpy as np
import cvxpy as cp

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
        
        if self.verbose:
            print('Optimal Value: ' + str(self.opt_fx))
            print('Optimal Point: ' + str(self.opt_x))
    
    def step(self):
        if self.verbose:
            print('iter: ' + str(self.cur_iter))
    
    def stop_cond(self):
        return False
    
        
class ProxBundle(OptAlg):
    def __init__(self, objective, max_iter = 10, x0 = None):
        super(ProxBundle,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.constraints    = []
        self.p              = cp.Variable(self.x_dim) # variable of optimization
        self.v              = cp.Variable() # value of cutting plane model
        
        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.update_bundle()

    def step(self):
        
        super(ProxBundle,self).step()
        
        prox_objective = self.v + cp.power(cp.norm(self.p - self.cur_x,2),2)
        
        prob = cp.Problem(cp.Minimize(prox_objective),self.constraints)
        prob.solve(solver='CVXOPT',abstol=1e-10, maxiter=100, verbose=True)
        
        # Update current iterate value and update the bundle
        self.cur_x      = self.p.value
        self.update_bundle()
        
        # Update paths and bundle constraints
        self.cur_iter    += 1
    
    def update_bundle(self):
        
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
        
        
    