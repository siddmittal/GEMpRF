import numpy as np

NUM_DIMENSIONS = 4
NUM_EXTRA_DIMENSIONS  =1
NUM_RANDOM_DIMENSIONS = 2

neighbours_xy = np.array([[1.1, 2.2], [3.1, 4.2], [2.1, 2.2], [6.1, 7.2]])
num_xy_neighbours = len(neighbours_xy)

sigma_major_values = np.array([100, 200, 300])
sigma_minor_values = np.array([33, 44])

total_neighbours = num_xy_neighbours * len(sigma_major_values) * len(sigma_minor_values)


neighbours = np.zeros((total_neighbours, NUM_DIMENSIONS))

counter = 0
for major_idx in range(len(sigma_major_values)):
    for minor_idx in range(len(sigma_minor_values)):
        for xy_point_idx in range(len(neighbours_xy)):
            major_dim_idx = 2
            minor_dim_idx = 3
            neighbours[counter, 0:NUM_RANDOM_DIMENSIONS] =  neighbours_xy[xy_point_idx]
            neighbours[counter, major_dim_idx] = sigma_major_values[major_idx]
            neighbours[counter, minor_dim_idx] = sigma_minor_values[minor_idx]
            counter = counter + 1

print(neighbours)        


