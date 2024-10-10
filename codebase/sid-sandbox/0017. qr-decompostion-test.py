import numpy as np
import matplotlib.pyplot as plt

# Define the 2D array
# data = np.array([
#     [12, -51, 4],
#     [6, 167, -68],
#     [-4, 24, -41],
#     [-1, 1, 0],
#     [2, 0, 3]
# ], dtype=float)  # specify data type as 'float' to match C++ double

# # Define the 2D array
# data = np.array([
#     [1, 1, 2],
#     [1, 1, 0],
#     [1, 0, 0]
# ], dtype=float)  # specify data type as 'float' to match C++ double

# data = np.array([
#     [1.00012016, 1.00012016],
#     [1.00012016, 1.00012016],    
# ], dtype=float)  # specify data type as 'float' to match C++ double

data = np.array([
    [1.16554,  0,  1.16554,  0.00641622],
    [1.16554,  0,  1.16554,  0.00641622],    
], dtype=float)  # specify data type as 'float' to match C++ double

q, r = np.linalg.qr(data)
print(q)
print(r)

