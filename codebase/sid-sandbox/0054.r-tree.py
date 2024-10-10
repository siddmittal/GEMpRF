from rtree import index
from random import uniform

# Function to generate random 3D points
def generate_random_point():
    return (uniform(-9, 9), uniform(-7, 7), uniform(0.5, 5))

# Generate random 3D points
num_points = 600000
points = [(i, generate_random_point()) for i in range(num_points)]

# Build R-tree index
p = index.Property()
p.dimension = 3
idx = index.Index(properties=p)
for i, point in points:
    idx.insert(i, point)

# Function to query for the nearest points
def query_nearest_points(query_point, k=27):
    results = idx.nearest(query_point, k)
    nearest_points = [points[result][1] for result in results]
    return nearest_points

# Example query
query_point = (2, 3, 4)  # Example query point
nearest_points = query_nearest_points(query_point)
print("27 Nearest points to query point:")
for point in nearest_points:
    print(point)
