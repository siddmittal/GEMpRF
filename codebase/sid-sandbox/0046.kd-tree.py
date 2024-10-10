import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import numpy as np

# # Generate some sample integer points in 3D
# points = np.array([[1, 2, 3],
#                    [5, 6, 7],
#                    [3, 8, 9],
#                    [9, 4, 5],
#                    [7, 5, 2]])



# # Define ranges for X, Y, and Z values
# x_values = np.linspace(-9, 9, 4)
# y_values = np.linspace(-10, 10, 3)
# z_values = np.linspace(0, 5, 2)

# Define ranges for X, Y, and Z values
x_values = np.linspace(-9, 9, 5)
y_values = np.linspace(-10, 10, 3)
z_values = np.linspace(0, 5, 5)

# Create meshgrid to generate combinations of X, Y, and Z values
X, Y, Z = np.meshgrid(x_values, y_values, z_values)

# Reshape the meshgrid arrays to create a single array of all combinations
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Create a KDTree object with the sample points
kdtree = KDTree(points, leaf_size=30, metric='euclidean')

# Query point for which nearest neighbors will be found
# query_point = np.array([[4, 5, 6]])
query_point = np.array([[4.5, 5.2, 3.6]])

# Find the indices of the nearest neighbors and their distances
distances, indices = kdtree.query(query_point, k=4)

# Plot all points in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='All Points')

# Plot the query point in 3D
ax.scatter(query_point[:, 0], query_point[:, 1], query_point[:, 2], color='red', label='Query Point')

# Plot the nearest neighbors in 3D
nearest_neighbors = points[indices[0]]  # Get nearest neighbors using indices
ax.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], nearest_neighbors[:, 2], color='green', label='Nearest Neighbors')

# Connect the query point with its nearest neighbors in 3D
for neighbor in nearest_neighbors:
    ax.plot([query_point[0, 0], neighbor[0]], [query_point[0, 1], neighbor[1]], [query_point[0, 2], neighbor[2]], color='gray', linestyle='--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('KD-Tree Nearest Neighbors in 3D')
ax.legend()
plt.show()

print()


###########################------------------2D version------------------#############

# import matplotlib.pyplot as plt
# from sklearn.neighbors import KDTree
# import numpy as np

# # Generate some sample integer points
# points = np.array([[1, 2],
#                    [5, 6],
#                    [3, 8],
#                    [9, 4],
#                    [7, 5]])

# # Create a KDTree object with the sample points
# kdtree = KDTree(points, leaf_size=30, metric='euclidean')

# # Query point for which nearest neighbors will be found
# query_point = np.array([[4, 5]])

# # Find the indices of the nearest neighbors and their distances
# distances, indices = kdtree.query(query_point, k=2)

# # Plot all points
# plt.figure(figsize=(8, 6))
# plt.scatter(points[:, 0], points[:, 1], color='blue', label='All Points')

# # Plot the query point
# plt.scatter(query_point[:, 0], query_point[:, 1], color='red', label='Query Point')

# # Plot the nearest neighbors
# nearest_neighbors = points[indices[0]]  # Get nearest neighbors using indices
# plt.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], color='green', label='Nearest Neighbors')

# # Connect the query point with its nearest neighbors
# for neighbor in nearest_neighbors:
#     plt.plot([query_point[0, 0], neighbor[0]], [query_point[0, 1], neighbor[1]], color='gray', linestyle='--')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('KD-Tree Nearest Neighbors')
# plt.legend()
# plt.grid(True)
# plt.show()

# print()
