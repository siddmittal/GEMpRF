import numpy as np
from numba import njit

@njit
def get_block_indices_with_Sigma(row, col, frame, nCols, nRows, nFrames, distance):
    _x_from = max(0, col - distance)
    _x_to = min(nCols - 1, col + distance)

    _y_from = max(0, row - distance)
    _y_to = min(nRows - 1, row + distance)

    _z_from = max(0, frame - distance)
    _z_to = min(nFrames - 1, frame + distance)

    shape = ((_x_to + 1) - _x_from) * ((_y_to + 1) - _y_from) * ((_z_to + 1) - _z_from)
    out = np.empty(shape=(shape, 3), dtype=np.int32)

    i = 0
    for _x in range(_x_from, _x_to + 1):
        for _y in range(_y_from, _y_to + 1):
            for _z in range(_z_from, _z_to + 1):
                out[i, 0] = _x
                out[i, 1] = _y
                out[i, 2] = _z
                i += 1
    return out

@njit
def get_block_mean_pairs(row, col, frame, mu_X, mu_Y, sigma_grid, distance=1):
    nRows = mu_X.shape[0]
    nCols = mu_X.shape[1]
    nFrames = mu_X.shape[2]

    # NOTE: row & col arguments are FLIPPED
    block_indices = get_block_indices_with_Sigma(col=row, row=col, frame=frame, nRows=nRows, nCols=nCols, nFrames=nFrames, distance=1)

    # Create arrays to store block_mu_x, block_mu_y, and block_sigma
    block_mu_x = np.empty(block_indices.shape[0], dtype=mu_X.dtype)
    block_mu_y = np.empty(block_indices.shape[0], dtype=mu_Y.dtype)
    block_sigma = np.empty(block_indices.shape[0], dtype=sigma_grid.dtype)

    # Populate block_mu_x, block_mu_y, and block_sigma using loop-based indexing
    for i in range(block_indices.shape[0]):
        r, c, f = block_indices[i]
        block_mu_x[i] = mu_X[r, c, f]
        block_mu_y[i] = mu_Y[r, c, f]
        block_sigma[i] = sigma_grid[r, c, f]

    gaussian_args = np.vstack((block_mu_x, block_mu_y, block_sigma)).T

    return gaussian_args, block_indices

@njit
def Grids2MpInv_numba(mu_X_grid, mu_Y_grid, sigma_grid):
    nRows = mu_X_grid.shape[0]
    nCols = mu_X_grid.shape[1]
    nFrames = mu_X_grid.shape[2]
    arr_2d_location_inv_M = [] # np.empty((nRows * nCols * nFrames), dtype=object)

    # linear equations related
    num_unknown_coefficients = 10
    num_linear_equations = 4

    for frame in range(nFrames):
        for row in range(nRows):
            for col in range(nCols):        
                gaussian_args, _ = get_block_mean_pairs(row, col, frame, mu_X_grid, mu_Y_grid, sigma_grid, distance=1)   
                num_neighbours = len(gaussian_args)                                                     
                M = np.zeros((num_neighbours * num_linear_equations, num_unknown_coefficients), dtype=float)
                for i in  range(len(gaussian_args)):
                    vecX = gaussian_args[i]
                    x = np.array(
                        [[vecX[0] ** 2  ,   vecX[1] ** 2    , vecX[2] ** 2  ,   2. * vecX[0] * vecX[1]      ,   2. * vecX[0] * vecX[2]      ,    2. * vecX[1] * vecX[2]     ,   vecX[0]     ,   vecX[1]     ,   vecX[2] ,   1],
                         [2. * vecX[0]  ,   0               , 0             ,   2. * vecX[1]                ,   2. * vecX[2]                ,    0                          ,   1           ,   0           ,   0       ,   0],
                         [0             ,   2. * vecX[1]    , 0             ,   2. * vecX[0]                ,   0                           ,    2. * vecX[2]               ,   0           ,   1           ,   0       ,   0],
                         [0             ,   0               , 2. * vecX[2]  ,   0                           ,   2. * vecX[0]                ,    2. * vecX[1]               ,   0           ,   0           ,   1       ,   0]]
                         )
                    M[i * num_linear_equations : (i * num_linear_equations) + num_linear_equations, :] = x

                Mp_inv = np.linalg.pinv(M)                                
                arr_2d_location_inv_M.append(Mp_inv)

    return arr_2d_location_inv_M


# @njit
def Weighted_Grids2MpInv_numba(mu_X_grid, mu_Y_grid, sigma_grid, Weights):
    nRows = mu_X_grid.shape[0]
    nCols = mu_X_grid.shape[1]
    nFrames = mu_X_grid.shape[2]
    arr_2d_location_inv_M = [] # np.empty((nRows * nCols * nFrames), dtype=object)

    # linear equations related
    num_unknown_coefficients = 10
    num_linear_equations = 4

    for frame in range(nFrames):
        for row in range(nRows):
            for col in range(nCols):        
                gaussian_args, block_indices = get_block_mean_pairs(row, col, frame, mu_X_grid, mu_Y_grid, sigma_grid, distance=1)   
                num_neighbours = len(gaussian_args)                                                     
                M = np.zeros((num_neighbours * num_linear_equations, num_unknown_coefficients), dtype=float)
                weighted_M_pInv = np.zeros((num_neighbours * num_linear_equations, num_unknown_coefficients), dtype=float)
                # W = np.zeros((num_neighbours * Weights.shape[1], Weights.shape[1]), dtype=float) # Weights.shape[1] provides the information about number of weights for each signal e.g. (Wx, Wy, Wsigma, Wsignal) in case of Gaussian model
                W = np.zeros((num_neighbours * num_linear_equations, num_neighbours * num_linear_equations), dtype=float)
                for i in  range(num_neighbours):
                    vecX = gaussian_args[i]
                    x = np.array(
                        [[vecX[0] ** 2  ,   vecX[1] ** 2    , vecX[2] ** 2  ,   2. * vecX[0] * vecX[1]      ,   2. * vecX[0] * vecX[2]      ,    2. * vecX[1] * vecX[2]     ,   vecX[0]     ,   vecX[1]     ,   vecX[2] ,   1],
                         [2. * vecX[0]  ,   0               , 0             ,   2. * vecX[1]                ,   2. * vecX[2]                ,    0                          ,   1           ,   0           ,   0       ,   0],
                         [0             ,   2. * vecX[1]    , 0             ,   2. * vecX[0]                ,   0                           ,    2. * vecX[2]               ,   0           ,   1           ,   0       ,   0],
                         [0             ,   0               , 2. * vecX[2]  ,   0                           ,   2. * vecX[0]                ,    2. * vecX[1]               ,   0           ,   0           ,   1       ,   0]]
                         ) 
                                       
                    M[i * num_linear_equations : (i * num_linear_equations) + num_linear_equations, :] = x
                    signal_flat_idx = (block_indices[i][2]*nRows*nCols) + (block_indices[i][0] * nCols) + block_indices[i][1] # (frame*nRows*nCols) + (row * nCols) + col
                    W[i * num_linear_equations : (i * num_linear_equations) + num_linear_equations, i * num_linear_equations : (i * num_linear_equations) + num_linear_equations] = Weights[signal_flat_idx] * np.eye(Weights.shape[1])
                
                # using Weighted least squares
                M_pInv_new = np.linalg.solve(M.T @ W @ M, M.T@W)
                    
                # orginal
                Mp_inv = np.linalg.pinv(M)                                
                # arr_2d_location_inv_M.append(Mp_inv)

                # weighted
                arr_2d_location_inv_M.append(M_pInv_new)

    return arr_2d_location_inv_M


###############################################----------------------------###########################################

class ParallelComputedCoefficientMatrix:    
    @classmethod
    def Wrapper_Grids2MpInv_numba(cls, mu_X_grid, mu_Y_grid, sigma_grid):
        arr_2d_location_inv_M = Grids2MpInv_numba(mu_X_grid, mu_Y_grid, sigma_grid)

        return arr_2d_location_inv_M
    
    @classmethod
    def Wrapper_Weighted_Grids2MpInv_numba(cls, mu_X_grid, mu_Y_grid, sigma_grid, Weights):
        arr_2d_location_inv_M = Weighted_Grids2MpInv_numba(mu_X_grid, mu_Y_grid, sigma_grid, Weights)
        return arr_2d_location_inv_M    
    
    @classmethod
    def Wrapper_get_block_indices_with_Sigma(cls, row, col, frame, nCols, nRows, nFrames, distance):
        # NOTE: row & col arguments are FLIPPED
        result = get_block_indices_with_Sigma(row=col, col=row, frame=frame, nCols=nCols, nRows=nRows, nFrames=nFrames, distance=distance)

        return result
