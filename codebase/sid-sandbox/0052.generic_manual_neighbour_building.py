import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# Given values
x_values = np.linspace(-10, 10, 5)
y_values = np.linspace(-10, 10, 5)
z_values = np.linspace(1, 5, 5)

# Create meshgrid to generate combinations of X, Y, and Z values
X, Y, Z = np.meshgrid(x_values, y_values, z_values)
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Define the query point
query_point = np.array([0, 0, 2])

# Find the indices of the nearest neighbors in each dimension
x_query_point_idx = np.argmin(np.abs(x_values - query_point[0]))
y_query_point_idx = np.argmin(np.abs(y_values - query_point[1]))
z_query_point_idx = np.argmin(np.abs(z_values - query_point[2]))

x_neighbours_indices = range(max(0, x_query_point_idx - 1), min(len(x_values), x_query_point_idx + 2))
y_neighbours_indices = range(max(0, y_query_point_idx - 1), min(len(y_values), y_query_point_idx + 2))
z_neighbours_indices = range(max(0, z_query_point_idx - 1), min(len(z_values), z_query_point_idx + 2))

# Preallocate an array for neighbor points
max_neighbours = len(x_neighbours_indices) * len(y_neighbours_indices) * len(z_neighbours_indices)
neighbours_points = np.full((max_neighbours, 3), np.nan)

# Compute and store the neighbor points without using append
index = 0
for x_idx in x_neighbours_indices:
    for y_idx in y_neighbours_indices:
        for z_idx in z_neighbours_indices:
            neighbours_points[index] = [x_values[x_idx], y_values[y_idx], z_values[z_idx]]
            index += 1

print(neighbours_points)


# Plot all points in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='All Points')

# Plot the query point in 3D
ax.scatter(query_point[0], query_point[1], query_point[2], color='red', label='Query Point')

# Plot the nearest neighbors in 3D
nearest_neighbors = neighbours_points  # Get nearest neighbors using indices
ax.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], nearest_neighbors[:, 2], color='green', label='Nearest Neighbors')

# Connect the query point with its nearest neighbors in 3D
for neighbor in nearest_neighbors:
    ax.plot([query_point[0], neighbor[0]], [query_point[1], neighbor[1]], [query_point[2], neighbor[2]], color='gray', linestyle='--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('KD-Tree Nearest Neighbors in 3D')
ax.legend()
plt.show()

print