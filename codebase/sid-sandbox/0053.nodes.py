import numpy as np

class Node:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.connected_nodes = set()

class Network:
    def __init__(self, nodes, grid_size):
        self.nodes = nodes
        self.grid_size = grid_size
        self.grid = {}
    
    def build_grid(self):
        for node in self.nodes:
            cell = tuple(np.floor(node.position / self.grid_size).astype(int))
            if cell not in self.grid:
                self.grid[cell] = set()
            self.grid[cell].add(node.id)
    
    def connect_adjacent_nodes(self):
        for cell_id in self.grid:
            adjacent_cells = self.get_adjacent_cells(cell_id)
            for adj_cell_id in adjacent_cells:
                if adj_cell_id in self.grid:
                    nodes_in_cell = self.grid[cell_id]
                    nodes_in_adj_cell = self.grid[adj_cell_id]
                    for node_id in nodes_in_cell:
                        for adj_node_id in nodes_in_adj_cell:
                            if node_id != adj_node_id:
                                self.nodes[node_id].connected_nodes.add(adj_node_id)
                                self.nodes[adj_node_id].connected_nodes.add(node_id)
    
    def get_adjacent_cells(self, cell_id):
        adjacent_cells = []
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    adj_cell_id = (cell_id[0] + dx, cell_id[1] + dy, cell_id[2] + dz)
                    adjacent_cells.append(adj_cell_id)
        return adjacent_cells
    
    def get_adjacent_nodes(self, query_node_id):
        return self.nodes[query_node_id].connected_nodes

# Generate 300,000 random 3D nodes
num_nodes = 300000
nodes = [Node(i, np.random.rand(3)) for i in range(num_nodes)]

# Build the network
network = Network(nodes, grid_size=0.1)  # Adjust the grid size as needed
network.build_grid()
network.connect_adjacent_nodes()

# Select a random node to query
query_node_id = np.random.randint(0, num_nodes)
adjacent_nodes = network.get_adjacent_nodes(query_node_id)

print(f"Adjacent nodes to node {query_node_id}: {adjacent_nodes}")





##################################################################################################

# import numpy as np

# class Node:
#     def __init__(self, id, position):
#         self.id = id
#         self.position = np.array(position)
#         self.connected_nodes = set()

# class Network:
#     def __init__(self):
#         self.nodes = {}
    
#     def add_node(self, node):
#         self.nodes[node.id] = node
    
#     def add_connection(self, node_id_1, node_id_2):
#         self.nodes[node_id_1].connected_nodes.add(node_id_2)
#         self.nodes[node_id_2].connected_nodes.add(node_id_1)
    
#     def get_immediate_connections(self, query_node_id):
#         return self.nodes[query_node_id].connected_nodes

# # Generate 300,000 random 3D nodes
# num_nodes = 300000
# nodes = [Node(i, np.random.rand(3)) for i in range(num_nodes)]

# # Build the network
# network = Network()
# for node in nodes:
#     network.add_node(node)

# # Assume we have connections between nodes (for demonstration, let's create some random connections)
# for _ in range(10 * num_nodes):  # Adjust the number of connections as needed
#     node_id_1 = np.random.randint(0, num_nodes)
#     node_id_2 = np.random.randint(0, num_nodes)
#     if node_id_1 != node_id_2:  # Avoid self-connections
#         network.add_connection(node_id_1, node_id_2)

# # Select a random node to query
# query_node_id = np.random.randint(0, num_nodes)
# immediate_connections = network.get_immediate_connections(query_node_id)

# print(f"Immediate connections to node {query_node_id}: {immediate_connections}")
