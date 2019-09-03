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
    
    def plot(self):
        pass
    
    # check that all optimization algorithms have the same objective and inputs
    def do_check(self):
        for alg in opt_algs:
        