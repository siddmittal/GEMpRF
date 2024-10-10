#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:52:52 2023

@author: mwoletz
"""

import numpy as np
import matplotlib.pyplot as plt

# dummy coeffient matrices (in the real program, we need to compute these matrices)
A = np.array([[1, 0.5], [0.5, 2]])
b = np.array([3, 4])
c = 3

# dummy mu_x and mu_y locations in a neighborhood
x,y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))

X = np.vstack((x.flatten(), y.flatten()))

def fun(x):
    return x.T @ A @ x + b @ x + c

def fun_dx(x):
    return 2. * A @ x + b

# Dummy function of vector 'X' (i.e. f(X)), which in our case is 'e' (i.e. (y.T@s')^2)
Xf = np.array([fun(x) for x in X.T])
Xf_dx = np.array([fun_dx(x) for x in X.T]).T # derivative of f(X)

def x2M(x):
    return np.array([[2. * x[0],   0,     2.*x[1], 1, 0],
                     [   0,      2.*x[1], 2.*x[0], 0, 1]])

M = np.vstack([x2M(x) for x in X.T])

# Neighbourhood matrix, will be different for each location on the grid
Mp = np.linalg.inv(M.T @ M) @ M.T # can be pre-computed

Xf_dx_flat = Xf_dx.T.flatten()

# Two possible Methods to estimate the coefficients matrices A and B (see document)
# method-1: Using linear least  square technique, but in this case, the inverses will be computed by the `lstsq()` for every input 'Xf_dx_flat' <<<<-----NOT IDEAL for us
ab_est, resid, rank, singular_values = np.linalg.lstsq(M, Xf_dx_flat, rcond=None)

# method-2: use the pre-computed matrices, simply multiply the 'Xf_dx_flat' with its CORRESPONDING 'Mp' matrix <<<<----BETTER approach
ab_est2 = Mp @ Xf_dx_flat

#def square2vec(i_s, j_s): squareform scipy pdist

print('done')
    
    
