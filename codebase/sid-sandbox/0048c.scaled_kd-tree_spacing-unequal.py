import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import numpy as np

# Define ranges for X, Y, and Z values
x_values = np.linspace(-0.3, 0.3, 5)
y_values = np.linspace(-0.3, 0.3, 5)
z_values = np.linspace(1, 6, 6) # unqual spacing (large spacing in z-direction)
# z_values = np.linspace(-0.3, 0.3, 5) # equal spacing case

# Create meshgrid to generate combinations of X, Y, and Z values
X, Y, Z = np.meshgrid(x_values, y_values, z_values)

# Reshape the meshgrid arrays to create a single array of all combinations
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Create a KDTree object with the rescaled points
points_mean = points.mean(axis=0)
centered_points = points - points_mean
points_std = centered_points.std(axis=0)
points_rescaled = centered_points / points_std
kdtree = KDTree(points_rescaled, leaf_size=30, metric='euclidean')

# Query point for which nearest neighbors will be found
# query_point = np.array([[0, 0, 0]]) # test query point for equal spacing
query_point = np.array([[0, 0, 2]]) # test query point for unequal spacing

# Find the indices of the nearest neighbors and their distances
distances, indices = kdtree.query((query_point - points_mean) / points_std, k=27)

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