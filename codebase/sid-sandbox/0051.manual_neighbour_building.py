import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import numpy as np


# Define ranges for X, Y, and Z values
# x_values = np.linspace(-0.3, 0.3, 10)
# y_values = np.linspace(-0.6, 0.6, 20)
# z_values = np.linspace(1, 6, 6) # unqual spacing (large spacing in z-direction)

x_values = np.linspace(-10, 10, 5)
y_values = np.linspace(-10, 10, 5)
z_values = np.linspace(1, 5, 5)

# 2D 
X, Y = np.meshgrid(x_values, y_values)
points_2d = np.column_stack((X.ravel(), Y.ravel()))

# Create meshgrid to generate combinations of X, Y, and Z values
X, Y, Z = np.meshgrid(x_values, y_values, z_values)
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


# 2D kd-tree
kdtree_2d = KDTree(points_2d, leaf_size=30, metric='euclidean')

query_points = np.array([[0, 0, 2],
                        [0.1, 0.03157895, 2.0],
                         [-0.1, 0.03157895, 2.0]])

query_point = query_points[0]

# x_query_point_idx = np.argwhere((x_values > 0.09) & (x_values < 0.12))[0][0]
x_query_point_idx = np.argwhere(np.isclose(x_values, query_point[0]))[0][0]
x_neighbours = x_values[x_query_point_idx - 1 : x_query_point_idx + 2]


y_query_point_idx = np.argwhere(np.isclose(y_values, query_point[1]))[0][0]
y_neighbours = y_values[y_query_point_idx - 1 : y_query_point_idx + 2]

z_query_point_idx = np.argwhere(np.isclose(z_values, query_point[2]))[0][0]
z_neighbours = z_values[z_query_point_idx - 1 : z_query_point_idx + 2]

# distances, indices = kdtree_2d.query(query_point[:, 0:2], k=8)
# neighbours = points_2d[indices[0]]

# neighbours
x, y, z = np.meshgrid(x_neighbours, y_neighbours, z_neighbours)
neighbours_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))


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
