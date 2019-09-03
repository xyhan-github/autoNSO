#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""

# Uses the prox-bundle method to solve the simple objective starting at (2,3)

#%%
import sys
sys.path.append('..')
from obj.objective import Objective
from algs.optAlg import ProxBundle
from vis.visualize import OptPlot

#%%
# f(x,y) = |x| + y^2
def simple2D(x):
    return abs(x[0]) + x[1]**2

# Create the objective function
Simple2D = Objective(simple2D)

# Run prox-bundle optimization algorithm
optAlg = ProxBundle(Simple2D, x0=[20,30], max_iter=100)
optAlg.optimize()

#%%
plot_lims = {'x1_max' : 40,
             'x2_max' : 40,
             'x1_min' : -40,
             'x2_min' : -40,}

opt_plot = OptPlot(opt_algs=optAlg, plot_lims=plot_lims)
opt_plot.plot()