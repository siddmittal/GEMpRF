# Define the size of the grid
size = 5
# Define the number of layers
num_layers = 4

# Loop through each layer
for layer in range(num_layers):
    # Create a new grid for each layer
    grid = [['-' for _ in range(size)] for _ in range(size)]
    
    # Display the current layer
    print(f"Layer {layer + 1}:")
    for row in grid:
        print(' '.join(row))
    
    print()  # Add a newline between layers

print()
