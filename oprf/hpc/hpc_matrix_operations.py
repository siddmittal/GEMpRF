import numpy as np


class MatrixOps:
    # indexing convention: (y, x, z) = (row, col, frame) = (uy, ux, sigma)
    @classmethod
    def flatIdx2ThreeDimIndices_old(cls, flatIdx, nRows, nCols, nFrames):
        flatIdx = np.atleast_1d(flatIdx)  # Convert to NumPy array to handle multiple values or a single value
        frame = flatIdx // (nRows * nCols)
        flatIdx = flatIdx % (nRows * nCols)
        row = flatIdx // nCols
        col = flatIdx % nCols
        return row, col, frame

    # indexing convention: (y, x, z) = (row, col, frame) = (uy, ux, sigma)
    @classmethod
    def flatIdx2ThreeDimIndices(cls, flatIdx, nRows, nCols, nFrames):        
        indices_3d = np.unravel_index(flatIdx, (nFrames, nRows, nCols), order='C')

        # result [frame, row, col] format
        indices_3d = np.column_stack(np.array(indices_3d))

        # Swap move first column to last to keep [row, col, frame] format
        indices_3d= np.roll(indices_3d, shift=-1, axis=1)        
        return indices_3d.reshape(-1) if len(indices_3d) == 1 else indices_3d

    @classmethod
    def threeDimIndices2FlatIdx(cls, threeDimIdx, nRows, nCols):
        #threeDimIdx = np.atleast_3d(threeDimIdx)  # Convert to NumPy array to handle multiple values or a single value
        row, col, frame = threeDimIdx.T  # Transpose to unpack rows and columns
        flatIdx = frame * (nRows * nCols) + (row * nCols + col)
        return flatIdx
