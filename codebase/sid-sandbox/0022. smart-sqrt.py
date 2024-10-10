#import numpy as np

## Create a 3x3 matrix and fill it with values from 1 to 9
#A = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9]])

## Calculate the transpose of the matrix
#transpose_matrix = np.transpose(A)

## Perform element-wise multiplication between rows and columns
#result_vector = (A**2).sum(axis=1)


## OR GENERIC WAY
#generic_result = np.diag(A@A.T)

## Print the result as a row vector
#print("Result as a Row Vector:")
#print(result_vector)
#print(generic_result)

###############################################################################################################
# import numpy as np

# my_array = np.array([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]])

# column_norms = np.linalg.norm(my_array, axis=0)
# normalized_array = my_array / column_norms

# print("Original Array:")
# print(my_array)
# print("\nNormalized Array:")
# print(normalized_array)

###############################################################################################################
# import numpy as np

# # Assuming you have normalized_array from the previous step
# normalized_array = np.array([[0.26726124, 0.32444284, 0.36514837],
#                              [1.06904497, 1.08277111, 1.09544512],
#                              [1.87082869, 1.84111339, 1.82574249]])

# # Generate Gaussian noise with mean 0 and std deviation 1 for each column
# noise = np.random.normal(0, 1, size=normalized_array.shape)

# # Add noise to each column separately
# noisy_normalized_array = normalized_array + noise

# print("Normalized Array:")
# print(normalized_array)
# print("\nNoisy Normalized Array:")
# print(noisy_normalized_array)


# import numpy as np

# A = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]]
#             )

# # Generate Gaussian noise with mean 0 and std deviation 1 for each column
# noise = np.random.normal(0, 1, size=A.shape)


import numpy as np

# Set the number of samples
num_samples = 5

# Generate two sets of samples with Gaussian noise (N(0, 1))
set1 = np.random.normal(0, 1, num_samples)
set2 = np.random.normal(0, 1, num_samples)

print("Set 1 (N(0, 1)):")
print(set1)

print("\nSet 2 (N(0, 1)):")
print(set2)
