#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:32:35 2019

@author: Xiaoyan
"""

from algs.optAlg import OptAlg

# We can also make an object that compares multiple paths
class OptPlot:
    def __init__(self,opt_algs=None):
        
        # List of optimization algorithms
        self.opt_algs = []
        self.x_dim    = None
        self.obj_func = None
        
        # Add to list of algorithms to compare at initialization
        if opt_algs is not None:
            self.add_alg(opt_algs)
            
    def add_alg(self,opt_algs):
        
        # Check it is an optimization algorithm
        if type(opt_algs) is not list:
            assert isinstance(opt_algs, OptAlg)
            opt_algs = [opt_algs]
        else:
            # Check every element is an optimization algorithm
            for alg in opt_algs:
                assert isinstance(opt_algs, OptAlg)
            
        # Add to list of optimization algorithms
        self.opt_algs += opt_algs
        
        # Do a check to make sure all algorithms match
        # After checking set the x_dim and obj_func to be the common one
        self.do_check()
    
    def plot2D(self):
        assert len(self.opt_algs) > 0
        assert self.x_dim == 1 # 1D domain for 2D plot
    
    def plot3D(self):
        assert len(self.opt_algs) > 0
        assert self.x_dim == 2 # 2D domain for 3D plot
    
    # check that all optimization algorithms have the same objective and inputs
    def do_check(self):
        
        # Use the first algorithm's dimension and objective as reference
        ref_alg = self.opt_algs[0]
        
        # Check that dimensions of input are all the same
        for alg in self.opt_algs:
            assert alg.x_dim == ref_alg.x_dim
            assert alg.objective.obj_func == ref_alg.objective.obj_func
        
        self.x_dim      = ref_alg.x_dim
        self.obj_func   = ref_alg.obj_func
        
        