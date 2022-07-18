#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""

# Uses the prox-bundle method to solve the simple objective starting at (2,3)

#%%
import sys
import numpy as np
sys.path.append('..')

from obj.obj_funcs import maxQuadratic
from algs.prox_bundle import ProxBundle

# Create the objective function
MaxFunc = maxQuadratic(n=10, k=5)

# Run prox-bundle optimization algorithm
optAlg = ProxBundle(MaxFunc, x0=np.ones(10)*0.1, max_iter=100, mu=10)
optAlg.optimize()