import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rtree import index
from random import uniform
import numpy as np
from sklearn.neighbors import KDTree

numX = 51
numY = 51
numSigma = 8
space_xx = np.linspace(-9, 9, numX)
space_yy = np.linspace(-9, 9, numY)
space_zz = np.linspace(0.5, 5, numSigma)
num_points = 100
NUM_VARIABLE_DIMENSIONS = 2 # dimensions that we consider random e.g. in case of 2D Gaussian model, we consider X and Y as variable dimensions (i.e. they can have random distribution)

additional_dimensions_dummy = [space_zz, space_xx] # NOTE: For all specified dimension of the pRF model, we will store the values of the dimension in a list to make rest of the program generic
__each_dimension_unique_points = [space_xx, space_yy, space_zz]
additional_dimensions = [space_zz]

def get_xy_points():
    xy_points =  [(generate_random_point()) for i in range(num_points)]
    xy_points = np.array(xy_points)
    return xy_points

def generate_random_point():
    return (uniform(-9, 9), uniform(-7, 7))    

def get_hyperplane_distribution(xy_points):
    z_values = [0.5, 1, 1.5, 2]
    multidimensional_points = [(x, y, z) for x, y in xy_points for z in z_values]
    return multidimensional_points



# Generate random 3D points
xy_points = get_xy_points()
kdtree = KDTree(xy_points, leaf_size=30, metric='euclidean')
multidimensional_points = get_hyperplane_distribution(xy_points)

# # Function to query for the nearest points
# def query_nearest_points(query_point, k=27):
#     results = idx.nearest(query_point, k)
#     nearest_points = [points[result][1] for result in results]
#     return nearest_points

# Example query
query_point = np.array([space_xx[numX//2], space_yy[numY//2], space_zz[numSigma//2]]) 
query_point_xy = np.array([[query_point[0], query_point[1]]])
distances_xy, indices_xy = kdtree.query(query_point_xy, k=9)
neighbours_xy = (xy_points[indices_xy])[0] ######################################<<<<-----------------------------
query_point_extra_dimensions_values = np.array([query_point[d] for d in np.arange(NUM_VARIABLE_DIMENSIONS, len(query_point))])

# gather neighbouring values in the fixed dimensions
neighbours_fixed_dimensions = []        
for i in np.arange(NUM_VARIABLE_DIMENSIONS, len(query_point)):
    query_point_coord_val = query_point[i]
    extra_dim_values = __each_dimension_unique_points[i]
    coord_idx = np.argmin(np.abs(extra_dim_values - query_point_coord_val))
    coord_neighbours = extra_dim_values[range(max(0, coord_idx - 1), min(len(extra_dim_values), coord_idx + 2))]
    neighbours_fixed_dimensions.append(coord_neighbours)

# covert 2D neighbours to multi-dimensionals points
num_extra_dimensions = len(__each_dimension_unique_points) - NUM_VARIABLE_DIMENSIONS
num_neighbours_per_extra_dimension = np.array([len(neighbours_fixed_dimensions[d]) for d in range(len(neighbours_fixed_dimensions))])
total_neighbours = len(neighbours_xy) * np.prod(num_neighbours_per_extra_dimension)
multi_dim_neighbours = np.zeros((total_neighbours, len(query_point)))

for extra_dim in range(num_extra_dimensions):
    num_elements_current_dim = len(additional_dimensions[extra_dim])
        for idx in range(num_elements_current_dim):
        m


for dim in range(len(neighbours_fixed_dimensions)):
    for idx  in range(len(neighbours_fixed_dimensions[dim])):
        i = 0


# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all points
x_points = [point[0] for point in multidimensional_points]
y_points = [point[1] for point in multidimensional_points]
z_points = [point[2] for point in multidimensional_points]
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