import numpy as np

# Function to add N 3x4 arrays element-wise
def add_elementwise(arrays):
    # Convert the list of arrays to a 3D NumPy array (N x 3 x 4)
    arrays_np = np.array(arrays)
    
    # Use numpy.sum along the first axis to perform element-wise addition
    result_array_np = np.sum(arrays_np, axis=0)
    
    return result_array_np

# Example usage
# Assuming 'array_of_arrays' is a list of N 3x4 arrays
array1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
array2 = np.array([[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24]])
array3 = np.array([[3, 6, 9, 12], [15, 18, 21, 24], [27, 30, 33, 36]])

array_of_arrays = [array1, array2, array3]

result_array = add_elementwise(array_of_arrays)

# Print the result
print("Resultant Array:")
print(result_array)
