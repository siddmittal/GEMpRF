# import numpy as np

# # Define your column vector of flat indices
# test_flat_indices = np.array([[9058], [3957], [8405], [9161], [11105], [969]])

# # Define the shape of your 3D array
# num_rows = 51
# num_cols = 51
# num_frames = 8

# # Use numpy.unravel_index to convert flat indices to 3D indices
# indices_3d = np.unravel_index(test_flat_indices, (num_frames, num_rows, num_cols), order='C')

# # result [frame, row, col] format
# indices_3d = np.column_stack(np.array(indices_3d)) #.T

# # Swap first and last columns to keep [row, col, frame] format
# indices_3d[:, [0, -1]] = indices_3d[:, [-1, 0]]

# print(indices_3d)

# print()


# import numpy as np

# # Create two 2D arrays filled with integer numbers
# array1 = np.random.randint(0, 100, size=(10, 3))
# print('array-1 before')
# print(array1)
# array2 = np.random.randint(0, 100, size=(10, 3))
# print('array-2 before')
# print(array2)

# # Create an index mask for indices 2, 4, and 7
# mask_indices = [2, 4, 7]

# # Replace values in the first array with values from the second array at the specified mask indices
# array1[mask_indices] = array2[mask_indices]

# print("Array 1 after replacement:")
# print(array1)

import numpy as np

# Create two 2D arrays filled with integer numbers
array1_list = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9],
               [10, 11, 12],
               [13, 14, 15],
               [16, 17, 18],
               [19, 20, 21],
               [22, 23, 24],
               [25, 26, 27],
               [28, 29, 30]]

array2 = np.random.randint(0, 100, size=(10, 3))

# Create an index mask for indices 2, 4, and 7
mask_indices = [2, 4, 7]

# Convert array1_list to a NumPy array
array1 = np.array(array1_list)

# Replace values in the first array with values from the second array at the specified mask indices
array1[mask_indices] = array2[mask_indices]

# If you want to convert the modified array1 back to a list
array1_list_modified = array1.tolist()

print("Array 1 (list) after replacement:")
print(array1_list_modified)
