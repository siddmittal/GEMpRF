import numpy as np
from numba import njit


@njit(nogil=True)
def GEM_Grids2MpInv_numba(multi_dim_points, multi_dim_points_neighbours):
    arr_2d_location_inv_M = [] # np.empty((nRows * nCols * nFrames), dtype=object)

    # linear equations specifications - Gaussian model
    num_unknown_coefficients = 10
    num_linear_equations = 4

    for multi_dim_point_idx in range(len(multi_dim_points)):
        multi_dim_point = multi_dim_points[multi_dim_point_idx]
        gaussian_args = multi_dim_points_neighbours[multi_dim_point_idx]
        num_neighbours = len(gaussian_args)                                                     
        M = np.zeros((num_neighbours * num_linear_equations, num_unknown_coefficients), dtype=float)
        for i in  range(len(gaussian_args)):
            vecX = gaussian_args[i]

            x = np.array(
                [[vecX[0] ** 2  ,   vecX[1] ** 2    , vecX[2] ** 2  ,   2. * vecX[0] * vecX[1]      ,   2. * vecX[0] * vecX[2]      ,    2. * vecX[1] * vecX[2]     ,   vecX[0]     ,   vecX[1]     ,   vecX[2] ,   1],    # e       
                    [2. * vecX[0]  ,   0               , 0             ,   2. * vecX[1]                ,   2. * vecX[2]                ,    0                          ,   1           ,   0           ,   0       ,   0], # de_dx   
                    [0             ,   2. * vecX[1]    , 0             ,   2. * vecX[0]                ,   0                           ,    2. * vecX[2]               ,   0           ,   1           ,   0       ,   0], # de_dy   
                    [0             ,   0               , 2. * vecX[2]  ,   0                           ,   2. * vecX[0]                ,    2. * vecX[1]               ,   0           ,   0           ,   1       ,   0]] # de_dsigma
                    )
            M[i * num_linear_equations : (i * num_linear_equations) + num_linear_equations, :] = x

        Mp_inv = np.linalg.pinv(M)                                
        arr_2d_location_inv_M.append(Mp_inv)

    return arr_2d_location_inv_M



class CoefficientMatix:
    @classmethod
    def Wrapper_Grids2MpInv_numba(cls, multi_dim_points, multi_dim_points_neighbours):
        arr_2d_location_inv_M = GEM_Grids2MpInv_numba(multi_dim_points, multi_dim_points_neighbours)

        return arr_2d_location_inv_M
    

