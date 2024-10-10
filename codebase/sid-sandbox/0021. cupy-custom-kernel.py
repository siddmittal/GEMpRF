import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Function to create a dummy image of a circle with specified center and radius
def create_circle_image(size, center, radius):
    image = np.zeros(size, dtype=np.float32)
    for x in range(size[0]):
        for y in range(size[1]):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                image[x, y] = 1.0
    return image

# Create four circle images
image1 = create_circle_image((1000, 1000), (250, 250), 250)
image2 = create_circle_image((1000, 1000), (750, 250), 250)
image3 = create_circle_image((1000, 1000), (250, 750), 250)
image4 = create_circle_image((1000, 1000), (750, 750), 250)

# Transfer the images to the GPU
gpu_image1 = cp.asarray(image1)
gpu_image2 = cp.asarray(image2)
gpu_image3 = cp.asarray(image3)
gpu_image4 = cp.asarray(image4)

# Create a custom kernel using raw_kernel
custom_kernel = cp.RawKernel(r'''
extern "C" __global__
void custom_kernel(float* a, float* b, float* c, float* d, float* result, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        result[idx] = a[idx] + b[idx] + c[idx] + d[idx];
    }
}
''', 'custom_kernel')

# Create an empty array on the GPU to store the result
result_image = cp.zeros((1000, 1000), dtype=cp.float32)

# Define grid and block dimensions
block_dim = (16, 16)
grid_dim = (result_image.shape[0] // block_dim[0], result_image.shape[1] // block_dim[1])

# Call the custom kernel to sum the images
custom_kernel(grid_dim, block_dim, (gpu_image1, gpu_image2, gpu_image3, gpu_image4, result_image, result_image.shape[0], result_image.shape[1]))

# Transfer the result back to the CPU
result_image_cpu = cp.asnumpy(result_image)

# Print or process the result as needed
print(result_image_cpu)
