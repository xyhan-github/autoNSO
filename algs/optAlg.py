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
    def __init__(self, objective, max_iter = 1000, x0 = None):
        
        if type(x0) is not np.ndarray:
            x0 = np.array(x0)
        
        assert len(x0.shape) == 1 # assert initial is vector
        
        self.max_iter   = 1000
        self.objective  = objective
        
        self.x0         = x0
        self.x_dim      = self.x0.shape[0]
        
        self.cur_iter  = 0
        self.cur_x     = None
        self.cur_fx    = None
        
        self.path_x      = np.array([])
        self.path_fx     = np.array([])
    
    def optimize(self):

        # Run the optimization algorithm until a stopping condition is hit
        stop = False
        for i in range(self.max_iter):
            self.step()
            stop = self.stop_cond()
            
            if stop:
                break
    
    def step(self):
        pass
    
    def stop_cond(self):
        return False
    
        
class ProxBundle(OptAlg):
    def __init__(self, objective, max_iter = 1000, x0 = None):
        super(ProxBundle,self).__init__(objective, max_iter = max_iter, x0 = x0)
        
        self.constraints    = []
        self.p              = cp.Variable(self.x_dim) # variable of optimization
        self.v              = cp.Variable() # value of cutting plane model
        
        # Add one bundle point to initial point
        self.cur_x = self.x0
        self.update_bundle()
        

    def step(self):
        
        prox_objective = self.v + cp.power(cp.norm(self.p - self.cur_x,2),2)
        
        prob = cp.Problem(cp.Minimize(prox_objective),self.constraints)
        prob.solve()
        
        # Update current iterate value and update the bundle
        self.cur_x      = self.p.value
        self.update_bundle()
        
        # Update paths and bundle constraints
        self.cur_iter    += 1
    
    def update_bundle(self):
        
        orcl_call           = self.objective.call_oracle(self.cur_x)
        self.cur_fx         = orcl_call['f']
        self.constraints    += [(self.cur_fx.copy() + 
                                 orcl_call(self.x0)['df']@(self.p - self.cur_x.copy())) <= self.v]
        self.path_x         = np.stack((self.path_x, self.cur_x))
        self.path_fx        = np.stack((self.path_fx, self.cur_fx))
        
        
    