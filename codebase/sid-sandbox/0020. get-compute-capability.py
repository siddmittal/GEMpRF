import pycuda.autoinit
import pycuda.driver as cuda

# Initialize CUDA device
device = cuda.Device(0)  # You can specify the device index if you have multiple GPUs

# Get the device attributes
attributes = device.get_attributes()

# Extract compute capability major and minor version
compute_capability = (attributes[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR], 
                      attributes[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR])

print("Compute Capability: {}.{}".format(*compute_capability))
