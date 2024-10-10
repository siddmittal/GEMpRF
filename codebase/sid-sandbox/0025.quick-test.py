# import numpy as np

# def get_neighbors(arr, row, col, distance=1):
#     # Define the indices for the neighbor pixels
#     nRows, nCols = arr.shape
#     r = np.linspace(row - distance, row + distance, 2 * distance + 1)
#     c = np.linspace(col - distance, col + distance, 2 * distance + 1)
#     nc, nr = np.meshgrid(c, r)
#     neighbors = np.vstack((nc.flatten(), nr.flatten())).T
    
#     # Filter out valid neighbor indices within the array bounds
#     valid_indices = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < nCols) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < nRows)
    
#     # Retrieve neighbor elements using NumPy indexing
#     neighbor_elements = arr[neighbors[valid_indices][:, 1].astype(int), neighbors[valid_indices][:, 0].astype(int)]
    
#     return neighbor_elements

# # Example usage with a 7x7 array:
# arr = np.array([[ 1,  2,  3,  4,  5,  6,  7],
#                 [ 8,  9, 10, 11, 12, 13, 14],
#                 [15, 16, 17, 18, 19, 20, 21],
#                 [22, 23, 24, 25, 26, 27, 28],
#                 [29, 30, 31, 32, 33, 34, 35],
#                 [36, 37, 38, 39, 40, 41, 42],
#                 [43, 44, 45, 46, 47, 48, 49]])

# row_coord = 1
# col_coord = 1
# neighbor_distance = 1  # You can adjust this distance to include more or fewer neighbors

# neighbors = get_neighbors(arr, row_coord, col_coord, neighbor_distance)
# print(neighbors)

###############################----------------------vectorized_transformation----------------------###########################
# import numpy as np

# def vectorized_transformation(current_tuples):
#     # Convert the input list of tuples to a NumPy array
#     current_tuples = np.array(current_tuples)

#     # Create two arrays for the transformation
#     arr1 = np.column_stack((current_tuples[:, 0],  current_tuples[:, 1]))
#     arr2 = np.column_stack((np.zeros(current_tuples.shape[0]), 3.0 * current_tuples[:, 0]))

#     # Stack the two arrays vertically to get the final result
#     result = np.vstack((arr1, arr2))

#     return result

# # Example usage:
# current_tuples = [(1, 2), (3, 4), (5, 6)]
# result = vectorized_transformation(current_tuples)
# print(result)


# import numpy as np

# def add_noisy_signals(signals, synthesis_ratio, noise_std=0.1):
#     """
#     Add noisy versions of signals along the columns of the 2D array.

#     Args:
#     signals (numpy.ndarray): 2D array of signals, where each column represents a signal.
#     synthesis_ratio (int): The number of noisy signals to generate for each original signal.
#     noise_std (float): Standard deviation of the added Gaussian noise.

#     Returns:
#     numpy.ndarray: 2D array containing the original signals and noisy signals.
#     """
#     num_signals, signal_length = signals.shape
#     noisy_signals = np.tile(signals, (1, synthesis_ratio + 1))
#     noise = np.random.normal(0, noise_std, size=(num_signals, synthesis_ratio * signal_length))
#     noisy_signals[:, signal_length:] += noise
    
#     return noisy_signals

# # Example usage:
# signal_length = 100
# signal_1 = np.random.rand(signal_length)  # Example original signal
# signal_2 = np.random.randn(signal_length)  # Example original signal
# signal_3 = np.random.randint(0, 10, signal_length)  # Example original signal
# signals = np.column_stack((signal_1, signal_2, signal_3))
# synthesis_ratio = 3  # Example: generate 3 noisy signals for each original signal

# noisy_signals = add_noisy_signals(signals, synthesis_ratio)

# # Print the resulting 2D array with noisy signals
# print(noisy_signals)

###############################--------------------------------------------###########################
# Python Program illustrating
# numpy.ravel() method

# import numpy as geek

# array = geek.arange(15).reshape(3, 5)
# print("Original array : \n", array)

# # Output comes like [ 0 1 2 ..., 12 13 14]
# # as it is a long output, so it is the way of
# # showing output in Python

# # About : 
# print("\nnumpy.ravel() : ", array.ravel())

# # Maintaining both 'A' and 'F' order
# print("\nMaintains A Order : ", array.ravel(order = 'F'))



#######################################################################################################################

# import numpy as np

# A = np.arange(15).reshape(3, 5)
# print("Original array : \n", A)

# At_A = (A ** 2).sum(axis=0)

# print("A.T@A: \n", At_A)

# B = np.ones([5, 5])
# print("B: \n", B)

# C = B * At_A[None:, ]
# print("C: \n", C)

import numpy as np

# Create a sample matrix
matrix = np.array([[1, 4, 9],
                  [16, 25, 36],
                  [49, 64, 81]])

# Calculate the element-wise power of (-1/2)
inv_sqrt_matrix = matrix ** (-1/2)

print(inv_sqrt_matrix)


