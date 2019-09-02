#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:22 2019

@author: Xiaoyan
"""
# Defines an objective
# Calls the oracle
# Oracle will output evaluation and gradient, unless told otherwise

import sys
sys.path.append('..')
from obj.objective import Objective

# f(x,y) = |x| + y^2
def simple2D(x):
    return abs(x[0]) + x[1]**2

Simple2D = Objective(simple2D)
out = Simple2D.call_oracle([1,2])

print(out)
# {'f': array(5., dtype=float32), 'df': array([1., 4.], dtype=float32)}
