import numpy as np

class CoefficientMatrix:
    @classmethod
    def create_cofficients_matrices_A_and_B(cls, coefficients):
        # coefficients = [ a11, a22, a33, a12, a13, a23, b1, b2,  b3] <----MIND that a22 is at second position
        '''   
            A =[[a11    a12     a13]
                [a12    a22     a23]
                [a13    a23     a33]]      
        ''' 
        A = np.array([
                [coefficients[0], coefficients[3], coefficients[4]],
                [coefficients[3], coefficients[1], coefficients[5]],
                [coefficients[4], coefficients[5], coefficients[2]]
            ])
    
        B = np.array([coefficients[6], coefficients[7], coefficients[8]])
        return A, B

    @classmethod
    def create_cofficients_matrices_A_B_and_C(cls, coefficients):
       # coefficients = [ a11, a22, a33, a12, a13, a23, b1, b2,  b3, C] <----MIND that a22 is at second position
       '''   
           A =[[a11    a12     a13]
               [a12    a22     a23]
               [a13    a23     a33]]      
       ''' 
       A = np.array([
               [coefficients[0], coefficients[3], coefficients[4]],
               [coefficients[3], coefficients[1], coefficients[5]],
               [coefficients[4], coefficients[5], coefficients[2]]
           ])
    
       B = np.array([coefficients[6], coefficients[7], coefficients[8]])

       C = coefficients[9]
       return A, B, C
    
    @classmethod
    def create_cofficients_matrices_A_B_and_C_vectorized(cls, coefficients):
       # coefficients = [ a11, a22, a33, a12, a13, a23, b1, b2,  b3, C] <----MIND that a22 is at second position
       '''   
           A =[[a11    a12     a13]
               [a12    a22     a23]
               [a13    a23     a33]]      
       ''' 
       # Extract each coefficient column
       a11 = coefficients[:, 0]
       a22 = coefficients[:, 1]
       a33 = coefficients[:, 2]
       a12 = coefficients[:, 3]
       a13 = coefficients[:, 4]
       a23 = coefficients[:, 5]

       # Stack them into A matrices for all frames
       A = np.stack([
           np.stack([a11, a12, a13], axis=1),
           np.stack([a12, a22, a23], axis=1),
           np.stack([a13, a23, a33], axis=1)
       ], axis=1)  # shape: (N, 3, 3)

       # B vectors for all frames
       B = coefficients[:, 6:9]  # shape: (N, 3)

       # C scalars for all frames
       C = coefficients[:, 9]     # shape: (N,)

       return A, B, C

    @classmethod
    def get_block_indices(cls, row, col, nRows, nCols, distance=1):
        # Define the indices for the neighbor pixels    
        r = np.linspace(row - distance, row + distance, 2 * distance + 1)
        c = np.linspace(col - distance, col + distance, 2 * distance + 1)
        nc, nr = np.meshgrid(c, r)
        neighbors = np.vstack((nc.flatten(), nr.flatten())).T
    
        # Filter out valid neighbor indices within the array bounds
        valid_indices = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < nCols) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < nRows)

        # Return the valid neighbor indices
        valid_neighbors = neighbors[valid_indices]
        return valid_neighbors

    @classmethod
    def get_block_indices_new(cls, row, col, nRows, nCols, distance=1):
        # Define the indices for the neighbor pixels    
        r = np.linspace(row - distance, row + distance, 2 * distance + 1)
        c = np.linspace(col - distance, col + distance, 2 * distance + 1)
        nc, nr = np.meshgrid(c, r)
        neighbors = np.vstack((nr.flatten(), nc.flatten())).T
    
        # Filter out valid neighbor indices within the array bounds
        valid_indices = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < nRows) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < nCols)

        # Return the valid neighbor indices
        valid_neighbors = neighbors[valid_indices]
        return valid_neighbors

    @classmethod
    def get_block_indices_with_Sigma(cls, row, col, frame, nRows, nCols, nFrames, distance=1):
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

    # Get neighbors (mu_x, mu_y) pairs
    @classmethod
    def _get_block_mean_pairs(cls, row, col, frame, mu_X, mu_Y, sigma_grid, distance=1):
        nRows = mu_X.shape[0]
        nCols = mu_X.shape[1]
        nFrames = mu_X.shape[2]

        block_indices = cls.get_block_indices_with_Sigma(row, col, frame, nRows, nCols, nFrames, distance=1)
    
        # Retrieve neighbor elements using NumPy indexing
        # block_mu_x = mu_X[block_indices[:, 1].astype(int), block_indices[:, 0].astype(int)]
        # block_mu_y = mu_Y[block_indices[:, 1].astype(int), block_indices[:, 0].astype(int)]

        block_mu_x = mu_X[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int), block_indices[:, 2].astype(int)]
        block_mu_y = mu_Y[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int), block_indices[:, 2].astype(int)]
        block_sigma = sigma_grid[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int), block_indices[:, 2].astype(int)]

        gaussian_args = (np.vstack((block_mu_x, block_mu_y, block_sigma))).T
    
        return gaussian_args

    @classmethod
    def _vecX2M(cls, vecX):            
        #########-----    a11               a22             a33                 a12                 a13                  a23                b1      b2      b3    
        return np.array([[2. * vecX[0]  ,   0               , 0             ,   2. * vecX[1]    ,   2. * vecX[2]    ,    0              ,   1   ,   0   ,   0],
                         [0             ,   2. * vecX[1]    , 0             ,   2. * vecX[0]    ,   0               ,    2. * vecX[2]   ,   0   ,   1   ,   0],
                         [0             ,   0               , 2. * vecX[2]  ,   0               ,   2. * vecX[0]    ,    2. * vecX[1]   ,   0   ,   0   ,   1]])

    @classmethod
    def _vecX2M_with_fx(cls, vecX):            
        #########-----    a11               a22             a33                 a12                 a13                  a23                b1      b2      b3      C    
        return np.array(
                        [[vecX[0] ** 2  ,   vecX[1] ** 2    , vecX[2] ** 2  ,   2. * vecX[0] * vecX[1]      ,   2. * vecX[0] * vecX[2]      ,    2. * vecX[1] * vecX[2]     ,   vecX[0]     ,   vecX[1]     ,   vecX[2] ,   1],
                         [2. * vecX[0]  ,   0               , 0             ,   2. * vecX[1]                ,   2. * vecX[2]                ,    0                          ,   1           ,   0           ,   0       ,   0],
                         [0             ,   2. * vecX[1]    , 0             ,   2. * vecX[0]                ,   0                           ,    2. * vecX[2]               ,   0           ,   1           ,   0       ,   0],
                         [0             ,   0               , 2. * vecX[2]  ,   0                           ,   2. * vecX[0]                ,    2. * vecX[1]               ,   0           ,   0           ,   1       ,   0]]
                         )

    @classmethod
    def Grids2MpInv(cls, mu_X_grid, mu_Y_grid, sigma_grid):
        nRows = mu_X_grid.shape[0]
        nCols = mu_X_grid.shape[1]
        nFrames = mu_X_grid.shape[2]
        arr_2d_location_inv_M = np.empty((nRows * nCols * nFrames), dtype=object)
        for frame in range(nFrames):
            for row in range(nRows):
                for col in range(nCols):        
                    gaussian_args = cls._get_block_mean_pairs(row, col, frame, mu_X_grid, mu_Y_grid, sigma_grid, distance=1)
                    #M = np.vstack([cls._vecX2M(x) for x in gaussian_args])
                    M = np.vstack([cls._vecX2M_with_fx(x) for x in gaussian_args])
                    Mp_inv = np.linalg.pinv(M)#.T @ M) @ M.T # Moore-Penrose Pseudoinverse (Mp_inv), (Mp_inv @ M = I)
                    flat_idx = frame * (nRows * nCols) + (row * nCols + col)
                    arr_2d_location_inv_M[flat_idx] = Mp_inv

        return arr_2d_location_inv_M



    @classmethod
    def Grids2MpInv_org(cls, mu_X_grid, mu_Y_grid):
        nRows = mu_X_grid.shape[0]
        nCols = mu_X_grid.shape[1]
        nFrame = mu_X_grid.shape[2]
        arr_2d_location_inv_M = np.empty((nRows * nCols), dtype=object)
        for row in range(nRows):
            for col in range(nCols):        
                block_mean_pairs = cls._get_block_mean_pairs(row, col, mu_X_grid, mu_Y_grid, distance=1)
                M = np.vstack([cls._vecX2M(x) for x in block_mean_pairs])
                Mp_inv = np.linalg.pinv(M)#.T @ M) @ M.T # Moore-Penrose Pseudoinverse (Mp_inv), (Mp_inv @ M = I)
                arr_2d_location_inv_M[row * nCols + col] = Mp_inv

        return arr_2d_location_inv_M
