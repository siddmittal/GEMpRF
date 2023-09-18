import numpy as np

# Test: for a given locaton (row, col), collecting values from all the slices of a 3d matrix 
# Define the dimensions of the 2D matrices
nRows = 3
nCols = 4

# Create a 3D matrix with 5 2D matrices
matrix_3d = np.random.randint(1, 11, size=(5, nRows, nCols))

# Print the 3D matrix
print(matrix_3d)

# Extract all values at (row, col) = (1, 2) from all slices of the 3D matrix
values_at_1_2 = matrix_3d[:, 1, 2]

# Print the collected values
print(values_at_1_2)