import numpy as np

def get_block_indices(row, col, nRows, nCols, distance=1):
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

def get_block_indices_new(row, col, nRows, nCols, distance=1):
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

# Get neighbors (mu_x, mu_y) pairs
def _get_block_mean_pairs(row, col, mu_X, mu_Y, distance=1):
    nRows = mu_X.shape[0]
    nCols = mu_X.shape[1]

    block_indices = get_block_indices_new(row, col, nRows, nCols, distance=1)
    
    # Retrieve neighbor elements using NumPy indexing
    # block_mu_x = mu_X[block_indices[:, 1].astype(int), block_indices[:, 0].astype(int)]
    # block_mu_y = mu_Y[block_indices[:, 1].astype(int), block_indices[:, 0].astype(int)]

    block_mu_x = mu_X[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int)]
    block_mu_y = mu_Y[block_indices[:, 0].astype(int), block_indices[:, 1].astype(int)]

    mean_pairs = (np.vstack((block_mu_x, block_mu_y))).T
    
    return mean_pairs



# # Get neighbors (mu_x, mu_y) pairs
# def _get_block_mean_pairs_WORKED(row, col, mu_X, mu_Y, distance=1):
#     nRows = mu_X.shape[0]
#     nCols = mu_X.shape[1]

#     # Define the indices for the neighbor pixels    
#     r = np.linspace(row - distance, row + distance, 2 * distance + 1)
#     c = np.linspace(col - distance, col + distance, 2 * distance + 1)
#     nc, nr = np.meshgrid(c, r)
#     neighbors = np.vstack((nc.flatten(), nr.flatten())).T
    
#     # Filter out valid neighbor indices within the array bounds
#     valid_indices = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < nCols) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < nRows)
    
#     # Retrieve neighbor elements using NumPy indexing
#     block_mu_x = mu_X[neighbors[valid_indices][:, 1].astype(int), neighbors[valid_indices][:, 0].astype(int)]
#     block_mu_y = mu_Y[neighbors[valid_indices][:, 1].astype(int), neighbors[valid_indices][:, 0].astype(int)]

#     mean_pairs = (np.vstack((block_mu_x, block_mu_y))).T
    
#     return mean_pairs

def _vecX2M(vecX):                    
    return np.array([[2. * vecX[0]  ,   0               ,   2. * vecX[1]  ,   1   ,   0],
                     [0             ,   2. * vecX[1]    ,   2. * vecX[0]  ,   0   ,   1]])

def Grids2MpInv(mu_X_grid, mu_Y_grid):
    nRows = mu_X_grid.shape[0]
    nCols = mu_X_grid.shape[1]
    arr_2d_location_inv_M = np.empty((nRows * nCols), dtype=object)
    for row in range(nRows):
        for col in range(nCols):        
            block_mean_pairs = _get_block_mean_pairs(row, col, mu_X_grid, mu_Y_grid, distance=1)
            M = np.vstack([_vecX2M(x) for x in block_mean_pairs])
            Mp_inv = np.linalg.pinv(M)#.T @ M) @ M.T # Moore-Penrose Pseudoinverse (Mp_inv), (Mp_inv @ M = I)
            arr_2d_location_inv_M[row * nCols + col] = Mp_inv

    return arr_2d_location_inv_M

##########################################---------program---------------------##############################
if __name__ == "__main__":        
    # ---------------------------------Variables
    search_space_rows = 101
    search_space_cols = 101

    # grid of Means
    search_space_yy = np.linspace(-9, +9, search_space_rows)
    search_space_xx = np.linspace(-9, +9, search_space_cols)
    mu_X_grid, mu_Y_grid = np.meshgrid(search_space_xx, search_space_yy) 

    arr_2d_location_inv_M = Grids2MpInv(mu_X_grid, mu_Y_grid)
            
    print('done')