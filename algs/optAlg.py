#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from IPython import embed
from numpy import format_float_scientific as ffs

class OptAlg:
    def __init__(self, objective, max_iter = 1000, x0 = None, verbose=True, switch_crit=None, save_multiple=False):
        assert x0 is not None

        if type(x0) is not np.ndarray:
            x0 = np.array(x0)
        assert len(x0.shape)==1  # x0 is a vector
        
        assert len(x0.shape) == 1 # assert initial is vector
        
        self.max_iter   = max_iter
        self.objective  = objective
        
        self.x0         = x0
        self.x_dim      = self.x0.shape[0]
        
        self.cur_iter  = 0
        self.cur_x     = None
        self.cur_fx    = None
        self.path_x      = None
        self.path_fx     = np.zeros((0,1))
        self.fx_step     = float('inf')
        
        self.verbose = verbose
        
        self.opt_x     = None
        self.opt_fx    = None
        self.total_iter = None
        
        self.name = None

        # function to trigger saving the checkpoint
        self.switch_crit = switch_crit
        self.saved_bundle = []
        self.save_multiple = save_multiple
    
    def optimize(self):

        # Run the optimization algorithm until a stopping condition is hit
        while self.cur_iter < self.max_iter:
            self.step()            
            if self.stop_cond():
                break

        self.opt_x = self.path_x[-1]
        self.opt_fx = self.path_fx[-1]
        self.total_iter = self.cur_iter
        
        if self.verbose:
            print('Optimal Value: ' + str(self.opt_fx))
            print('Optimal Point: ' + str(self.opt_x))
    
    def step(self):
        if self.verbose: # Prints previous step
            print('iter: ' + str(self.cur_iter) + ', obj: ' + str(self.cur_fx) + ', step: ' + str(ffs(self.fx_step,4)))
    
    def stop_cond(self):
        return False
    
    def update_params(self):
        if self.switch_crit is not None and self.switch_crit(self):
            if (len(self.saved_bundle)==0) or self.save_multiple:
                self.save_bundle()

    def save_bundle(self):
        pass