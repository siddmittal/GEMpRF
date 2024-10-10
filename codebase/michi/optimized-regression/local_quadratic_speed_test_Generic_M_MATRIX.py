#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:35:57 2024

@author: mwoletz
"""
import numpy as np
from time import perf_counter
from itertools import combinations, product
from sklearn.neighbors import KDTree
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numba as nb

search_space = {
  "visual_field": 13.5,
  "nRows": 151,
  "nCols": 151,
  "min_sigma": 0.5,
  "max_sigma": 5,
  "nSigma": 24
}

search_space_xx = np.linspace(-search_space["visual_field"], search_space["visual_field"], search_space["nCols"])
search_space_yy = np.linspace(-search_space["visual_field"], search_space["visual_field"], search_space["nRows"])
search_space_sigma_range = np.linspace(search_space["min_sigma"], search_space["max_sigma"], search_space["nSigma"]) # 0.5 to 1.5
mu_X_grid, mu_Y_grid, sigma_grid = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range) 

def get_block_indices_with_Sigma(row, col, frame, nRows, nCols, nFrames, distance=1):
    # Define the indices for the neighbor pixels    
    r = np.linspace(row - distance, row + distance, 2 * distance + 1)
    c = np.linspace(col - distance, col + distance, 2 * distance + 1)
    f = np.linspace(frame - distance, frame + distance, 2 * distance + 1)
    nc, nr, nf = np.meshgrid(c, r, f)
    neighbors = np.vstack((nr.flatten(), nc.flatten(), nf.flatten())).T

    # Filter out valid neighbor indices within the array bounds
    valid_indices = (neighbors[ :, 0] >= 0) & (neighbors[ :, 0] < nRows) & (neighbors[ :, 1] >= 0) & (neighbors[ :, 1] < nCols) & (neighbors[ :, 2] >= 0) & (neighbors[ :, 2] < nFrames) 

    # Return the valid neighbor indices
    valid_neighbors = neighbors[valid_indices]
    return valid_neighbors

def _get_block_mean_pairs(row, col, frame, mu_X, mu_Y, sigma_grid, distance=1):
    nRows = mu_X.shape[0]
    nCols = mu_X.shape[1]
    nFrames = mu_X.shape[2]

    block_indices = get_block_indices_with_Sigma(row, col, frame, nRows, nCols, nFrames, distance=1)

    # Retrieve neighbor elements using NumPy indexing
    # block_mu_x = mu_X[block_indices[:, 1].astype(int), block_indices[:, 0].astype(int)]
    # block_mu_y = mu_Y[block_indices[:, 1].astype(int), block_indices[:, 0].astype(int)]

    block_mu_x = mu_X[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int), block_indices[:, 2].astype(int)]
    block_mu_y = mu_Y[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int), block_indices[:, 2].astype(int)]
    block_sigma = sigma_grid[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int), block_indices[:, 2].astype(int)]

    gaussian_args = (np.vstack((block_mu_x, block_mu_y, block_sigma))).T

    return gaussian_args

def _vecX2M(vecX):            
    #########-----    a11               a22             a33                 a12                 a13                  a23                b1      b2      b3    
    return np.array([[2. * vecX[0]  ,   0               , 0             ,   2. * vecX[1]    ,   2. * vecX[2]    ,    0              ,   1   ,   0   ,   0],
                     [0             ,   2. * vecX[1]    , 0             ,   2. * vecX[0]    ,   0               ,    2. * vecX[2]   ,   0   ,   1   ,   0],
                     [0             ,   0               , 2. * vecX[2]  ,   0               ,   2. * vecX[0]    ,    2. * vecX[1]   ,   0   ,   0   ,   1]])

def _vecX2M_with_fx(vecX):            
    #########-----    a11               a22             a33                 a12                 a13                  a23                b1      b2      b3      C    
    return np.array(
                    [[vecX[0] ** 2  ,   vecX[1] ** 2    , vecX[2] ** 2  ,   2. * vecX[0] * vecX[1]      ,   2. * vecX[0] * vecX[2]      ,    2. * vecX[1] * vecX[2]     ,   vecX[0]     ,   vecX[1]     ,   vecX[2] ,   1],
                     [2. * vecX[0]  ,   0               , 0             ,   2. * vecX[1]                ,   2. * vecX[2]                ,    0                          ,   1           ,   0           ,   0       ,   0],
                     [0             ,   2. * vecX[1]    , 0             ,   2. * vecX[0]                ,   0                           ,    2. * vecX[2]               ,   0           ,   1           ,   0       ,   0],
                     [0             ,   0               , 2. * vecX[2]  ,   0                           ,   2. * vecX[0]                ,    2. * vecX[1]               ,   0           ,   0           ,   1       ,   0]]
                     )

#################################-------------------------------------------------------------------#################################
# NOTE: Michi's implementation to get M-Matrix in Generic Way
#################################-------------------------------------------------------------------#################################
@nb.njit
def getM(X):
    # Number of dimensions
    D = X.shape[1]
    
    # Number of points
    N = X.shape[0]
    
    # get all combinations
    combs = np.array([(j,i) for i in range(D) for j in range(i)]).T
    
    # number of combinations
    C = len(combs[0])
    
    n = N + D * N # numbers of rows in M
    m = 1 + D + D + C # constant + linear + quadratic + combinations
    
    # initialise M
    M = np.zeros((n,m))
    
    i = np.arange(n, step=1+D)
    
    # the quadratic terms
    M[i,:D] = X**2
    # the combinations
    M[i,D:D+C] = 2 * X[:,combs[0]] * X[:,combs[1]]
    # the linear terms
    M[i,D+C:-1] = X
    # the constant
    M[i,-1] = 1.
    
    # now do the derivatives
    for d in range(D):
        j = i+d+1
        # derivative of the constant
        M[j,d] = 2*X[:,d]
        
        # derivatives of the combinations
        # c_d = np.nonzero((combs[0] == d) | (combs[1] == d))[0]
        c_d_row, c_d = np.nonzero((combs == d))
        c_not_d = np.array([combs[1 - c_d_row[k], c_d[k]] for k in range(len(c_d))])
        # c_not_d = combs[:,c_d][combs[:,c_d] != d]
        # M[j[:,None],(c_d + D)[None,:]] = 2*X[:,c_not_d] # this is not supported in numba
        for c_d_i, c_not_d_i in zip(c_d, c_not_d):
            M[j, c_d_i + D] = 2*X[:,c_not_d_i]
        
        # derivative of the constant
        M[j, D+C+d] = 1.
        
    return M
    
# @profile
def Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid):
    nRows = mu_X_grid.shape[0]
    nCols = mu_X_grid.shape[1]
    nFrames = mu_X_grid.shape[2]
    arr_2d_location_inv_M = np.empty((nRows * nCols * nFrames), dtype=object)
    for frame in range(nFrames):
        for row in range(nRows):
            for col in range(nCols):        
                gaussian_args = _get_block_mean_pairs(row, col, frame, mu_X_grid, mu_Y_grid, sigma_grid, distance=1)
                #M = np.vstack([cls._vecX2M(x) for x in gaussian_args])
                M = np.vstack([_vecX2M_with_fx(x) for x in gaussian_args])
                Mp_inv = np.linalg.pinv(M)#.T @ M) @ M.T # Moore-Penrose Pseudoinverse (Mp_inv), (Mp_inv @ M = I)
                flat_idx = frame * (nRows * nCols) + (row * nCols + col)
                arr_2d_location_inv_M[flat_idx] = Mp_inv

    return arr_2d_location_inv_M

@nb.njit
def m_pinvM(M):
    return np.linalg.pinv(M)

@nb.jit(parallel=True, forceobj=True)
def m_pinvXs(Xs):
    Ms = [None] * len(Xs)
    for i in nb.prange(len(Xs)):
        Ms[i] = m_pinvM(getM(Xs[i]))
    return Ms

def m_pinv(xy, s):
    X = np.hstack((np.tile(xy, [len(s), 1]), np.repeat(s, len(xy))[:,None]))
    #X = np.hstack((np.vstack([xy]*len(s)), np.repeat(s, len(xy))[:,None]))
    return m_pinvM(getM(X))


# @profile
def grid_m_pinv(search_space_xx, search_space_yy, search_space_sigma_range):
    x,y = np.meshgrid(search_space_xx, search_space_yy, indexing='ij')
    X = np.vstack((x.flatten(), y.flatten())).T
    
    xy_tree = KDTree(X, metric='manhattan')
    
    xy_dist = max(search_space_xx[1] - search_space_xx[0], search_space_yy[1] - search_space_yy[0]) * 1.05
    xy_neighbors_i = xy_tree.query_radius(X, xy_dist)
    xy_neighbors = (X[n] for n in xy_neighbors_i)
    
    s_tree = KDTree(search_space_sigma_range[:,None], metric='manhattan')
    s_dist = (search_space_sigma_range[1] - search_space_sigma_range[0]) * 1.05
    
    s_neighbors_i = s_tree.query_radius(search_space_sigma_range[:,None], s_dist)
    s_neighbors = (search_space_sigma_range[n] for n in s_neighbors_i)

    xys = product(xy_neighbors, s_neighbors)
    
    # n_threads = os.cpu_count()
    # with ProcessPoolExecutor(max_workers=n_threads) as executor:
    #     Ms = executor.map(m_pinv, xys)
    # return list(Ms)
    Xs = [np.hstack((np.tile(xy, [len(s), 1]), np.repeat(s, len(xy))[:,None])) for xy, s in xys]
    return m_pinvXs(Xs)
    #return [m_pinv(xy,s) for xy, s in xys]
    
    
    
    

start = perf_counter()
M_sid = Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid)
stop = perf_counter()
t_sid = stop - start
print(f"Sid {t_sid}s...")
start = perf_counter()
M_grid = grid_m_pinv(search_space_xx, search_space_yy, search_space_sigma_range)
stop = perf_counter()
t_new = stop - start
print(f"new {t_new}s...")
print(f"relative {t_sid/t_new:.2%}")

row = 10
col = 12
frame = 3
gaussian_args = _get_block_mean_pairs(row, col, frame, mu_X_grid, mu_Y_grid, sigma_grid, distance=1)
M = np.vstack([_vecX2M_with_fx(x) for x in gaussian_args])
M2 = getM(gaussian_args)