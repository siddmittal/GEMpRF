import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rtree import index
from random import uniform
import numpy as np

numX = 10
numY = 10
numSigma = 8
MODE = "grid" # "random" 


# Function to generate random 3D points
def generate_grid_points(numX, numY, numSigma):
    X, Y, Z = np.meshgrid(np.linspace(-9, 9, numX), np.linspace(-9, 9, numY), np.linspace(0.5, 5, numSigma))
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    points = tuple(map(tuple, points))
    return points

def generate_random_point():
    return (uniform(-9, 9), uniform(-7, 7), uniform(0.5, 5))    

# Generate random 3D points
points_mean = 0
points_std = 1
if MODE == "grid":
    all_points = generate_grid_points(numX, numY, numSigma)
    points =  [(i, all_points[i]) for i in range(numX * numY * numSigma)]

    # z-score normalization
    points_mean = np.array(all_points).mean(axis=0)
    centered_points = np.array(all_points) - points_mean
    points_std = centered_points.std(axis=0)
    standardiszed_points = centered_points / points_std

elif MODE == "random":
    num_points = 100
    points =  [(i, generate_random_point()) for i in range(num_points)]

# Build R-tree index
p = index.Property()
p.dimension = 3
idx = index.Index(properties=p)
for i, point in points:
    if MODE=="grid":
        point = standardiszed_points[i]
    idx.insert(i, point)

# Function to query for the nearest points
def query_nearest_points(query_point, k=27):
    results = idx.nearest(query_point, k)
    nearest_points = [points[result][1] for result in results]
    return nearest_points

# Example query
# query_point = (2, 3, 4)  # Example query point
query_point = ((np.array([1, 1, 2.75]) ) - points_mean) / points_std
query_point = tuple(query_point)  # Example query point
nearest_points = query_nearest_points(query_point)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all points
x_points = [point[0] for _, point in points]
y_points = [point[1] for _, point in points]
z_points = [point[2] for _, point in points]
ax.scatter(x_points, y_points, z_points, c='b', label='All Points')

# Plot query point
ax.scatter(query_point[0], query_point[1], query_point[2], c='r', label='Query Point')

# Plot nearest points and draw lines
x_nearest = [point[0] for point in nearest_points]
y_nearest = [point[1] for point in nearest_points]
z_nearest = [point[2] for point in nearest_points]
ax.scatter(x_nearest, y_nearest, z_nearest, c='g', label='Nearest Points')

for i in range(len(nearest_points)):
    ax.plot([query_point[0], x_nearest[i]], [query_point[1], y_nearest[i]], [query_point[2], z_nearest[i]], c='g')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Queries Point and Nearest Neighbors')
plt.legend()
plt.show()


print