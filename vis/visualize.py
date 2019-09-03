#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:32:35 2019

@author: Xiaoyan
"""
import numpy as np
from algs.optAlg import OptAlg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from IPython import embed

# We can also make an object that compares multiple paths
class OptPlot:
    def __init__(self,opt_algs=None, plot_lims=None, axis_rot=(0,0)):
        
        # List of optimization algorithms
        self.opt_algs = []
        self.x_dim    = None
        self.obj_func = None
        
        # Add to list of algorithms to compare at initialization
        if opt_algs is not None:
            self.add_alg(opt_algs)
            
        self.x1_max = None
        self.x2_max = None
        self.x1_min = None
        self.x2_min = None
            
        if plot_lims is not None:
            for key in plot_lims:
                assert key in ['x1_max', 'x2_max', 'x1_min', 'x2_min']
                setattr(self, key, plot_lims[key])
        
        self.axis_rot = axis_rot
        plt.style.use('seaborn-white')
            
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
        
    def plot(self):
        assert self.x_dim is not None
        
        if self.x_dim == 1:
            self.plot2D()
        elif self.x_dim == 2:
            self.plot3D()
    
    def plot2D(self):
        assert len(self.opt_algs) > 0
        assert self.x_dim == 1 # 1D domain for 2D plot
    
    def plot3D(self):
        assert len(self.opt_algs) > 0
        assert self.x_dim == 2 # 2D domain for 3D plot
        
        # Set unset limits
        if None in [self.x1_max, self.x1_min, self.x2_max, self.x2_min]:
            x1_lim = float('-inf')
            x2_lim = float('-inf')
            
            for alg in self.opt_algs:
                path_max = np.max(np.abs(alg.path_x),axis=0)
                if path_max[0] > x1_lim:
                    x1_lim = path_max[0]
                if path_max[1] > x2_lim:
                    x2_lim = path_max[1]
            
            if self.x1_max == None:
                self.x1_max = 1.2 * x1_lim
            if self.x1_min == None:
                self.x1_min = -1.2 * x1_lim
            if self.x2_max == None:
                self.x2_max = 1.2 * x2_lim
            if self.x2_min == None:
                self.x2_min = -1.2 * x2_lim
        
        x1 = np.linspace(self.x1_min,self.x1_max,100)
        x2 = np.linspace(self.x2_min,self.x2_max,100)
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        
        f = lambda x1, x2: self.obj_func([x1,x2])
        vf = np.vectorize(f)
        fx_grid = vf(x1_grid, x2_grid)

        #Plot objective
        fig = plt.figure(figsize = (10,10))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1_grid,x2_grid,fx_grid,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.view_init(self.axis_rot[0], self.axis_rot[1])
        
        # Plot optimization path
        for alg in self.opt_algs:
            alg.path_x[:,0]
            ax.plot(alg.path_x[:,0],alg.path_x[:,1], alg.path_fx,
                    color = 'r', marker = '*', alpha = .4, label = alg.name)
        plt.legend()
        plt.show()

    # check that all optimization algorithms have the same objective and inputs
    def do_check(self):
        
        # Use the first algorithm's dimension and objective as reference
        ref_alg = self.opt_algs[0]
        
        # Check that dimensions of input are all the same
        # Check that there is actually a path that finished
        for alg in self.opt_algs:
            assert alg.x_dim == ref_alg.x_dim
            assert alg.objective.obj_func == ref_alg.objective.obj_func
            assert alg.opt_x is not None
        
        self.x_dim      = ref_alg.x_dim
        self.obj_func   = ref_alg.objective.obj_func
        
        