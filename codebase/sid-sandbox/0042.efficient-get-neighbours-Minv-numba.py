import numpy as np
import cProfile

from numba import njit, int32
from timeit import timeit


@njit
def get_arr(x, y, f, w, h, distance, frames):
    _x_from = max(0, x - distance)
    _x_to = min(w - 1, x + distance)

    _y_from = max(0, y - distance)
    _y_to = min(h - 1, y + distance)

    _z_from = max(0, f - distance)
    _z_to = min(frames - 1, f + distance)

    shape = ((_x_to + 1) - _x_from) * ((_y_to + 1) - _y_from) * ((_z_to + 1) - _z_from)
    out = np.empty((shape, 3), dtype=np.int32)

    i = 0
    for _x in range(_x_from, _x_to + 1):
        for _y in range(_y_from, _y_to + 1):
            for _z in range(_z_from, _z_to + 1):
                out[i, 0] = _x
                out[i, 1] = _y
                out[i, 2] = _z
                i += 1
    return out

class Neighbours:
    # Get neighbors
    @classmethod
    def get_neighbour_indices(cls, row, col, frame, distance=1):                
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

    @classmethod
    def MapIndexVsNeighbours(cls):
        neighbours_info = np.empty((nRows * nCols * nFrames), dtype=object)
        for frame in range(nFrames):
            for row in range(nRows):
                for col in range(nCols):        
                    neighbour_indices = cls.get_neighbour_indices(row, col, frame, distance=1)        
                    flat_idx = frame * (nRows * nCols) + (row * nCols + col)
                    neighbours_info[flat_idx] = neighbour_indices                            

        return neighbours_info
    


    @classmethod
    def MapIndexVsNeighbours_numba(cls, nRows, nCols, nFrames):
        neighbours_info = np.empty((nRows * nCols * nFrames), dtype=object)
        for frame in range(nFrames):
            for row in range(nRows):
                for col in range(nCols):
                    neighbour_indices = get_arr(row, col, frame, nRows, nCols, 1, nFrames)
                    flat_idx = frame * (nRows * nCols) + (row * nCols + col)
                    neighbours_info[flat_idx] = neighbour_indices
        return neighbours_info    


########################------------------main()-------##################
####--run
if __name__ == "__main__":  
    nRows = 151
    nCols = 151
    nFrames = 24
    # cProfile.run('Neighbours.MapIndexVsNeighbours()', sort='cumulative')  
    # cProfile.run('Neighbours.MapIndexVsNeighbours_numba(nRows, nCols, nFrames)', sort='cumulative')  

    
    t_numba = timeit("Neighbours.MapIndexVsNeighbours_numba(nRows, nCols, nFrames)", number=1, globals=globals())
    t_original = timeit("Neighbours.MapIndexVsNeighbours()", number=1, globals=globals())

    print(f"{t_numba=}")
    print(f"{t_original=}")

    print() 