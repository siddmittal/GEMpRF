import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class Neighbours:
    def __init__(self, points):
        self.__points = points

        # compute unique values for each dimension
        num_dimensions = points.shape[1]
        self.__each_dimension_unique_points = []
        for d in range(num_dimensions):
            self.__each_dimension_unique_points.append(np.unique(points[:, d]))        

    def get_adjacent_neighbours(self, query_point):        
        # Find the indices of the nearest neighbors in each dimension
        neighbours_indices = []        
        for values, query_coord in zip((self.unique_points[d] for d in range(self.num_dimensions)), query_point):
            idx = np.argmin(np.abs(values - query_coord))
            query_coord_neighbours = range(max(0, idx - 1), min(len(values), idx + 2))
            neighbours_indices.append(query_coord_neighbours)

        # Preallocate an array for neighbor points
        max_neighbours = np.prod([len(indices) for indices in neighbours_indices])
        neighbours_points = np.full((max_neighbours, self.num_dimensions), np.nan)

        # NOTE: Yes! we can use meshgrid here but to keep it Numba compatible, we are not using it
        # Compute and store the neighbor points without using append
        index = 0
        for indices in product(*neighbours_indices):
            neighbour_point = [values[idx] for values, idx in zip((self.unique_points[d] for d in range(self.num_dimensions)), indices)]
            neighbours_points[index] = neighbour_point
            index += 1

        return neighbours_points
    
    @property
    def points(self):
        return self.__points
    
    @property
    def unique_points(self):
        return self.__each_dimension_unique_points
    
    @property
    def num_dimensions(self):
        return len(self.unique_points)

def plot_neighbours(points, query_point, neighbours_points):
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


def get_dummy_points():
    x_values = np.linspace(-10, 10, 151)
    y_values = np.linspace(-10, 10, 151)
    z_values = np.linspace(1, 5, 16)

    # Create meshgrid to generate combinations of X, Y, and Z values
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    return points

# if __name__ == "__main__":
#     points = get_dummy_points()

#     num_dimensions = points.shape[1]

#     unique_points = []
#     for d in range(num_dimensions):
#         unique_points.append(np.unique(points[:, d]))

#     # Define the query point
#     query_point = np.array([0, 0, 2])

#     # get adjacent neighbours
#     nn = Neighbours(points=points)
#     neighbours_points = nn.get_adjacent_neighbours(query_point)

#     # test plot
#     plot_neighbours(points, query_point, neighbours_points)

#     print


###############################

if __name__ == "__main__":
    points = get_dummy_points()
    tri = Delaunay(points)

    neighbours = [np.unique(tri.simplices[np.any(np.isin(tri.simplices, i), 1)]) for i in range(points.shape[0])]

    print(neighbours)




    

    

