###########################################--------------Test Device Query-------------------------#########################################
import cupy as cp

# import pycuda
# import pycuda.driver as drv
# drv.init()

print ('CUDA device query (cuPy version)')

print( f'Detected {format(cp.cuda.runtime.getDeviceCount())} CUDA Capable device(s)')

for i in range(cp.cuda.runtime.getDeviceCount()):
    
    gpu_device = cp.cuda.Device(i)
    # print ('Device {}: {}'.format( i, gpu_device.name() ) )
    compute_capability = gpu_device.compute_capability #float( '%d.%d' % gpu_device.compute_capability )
    print ('Compute Capability: {}'.format(compute_capability))
    print ('Total Memory: {} megabytes'.format(gpu_device.mem_info[1]//(1024**2)))
    
    # The following will give us all remaining device attributes as seen 
    # in the original deviceQuery.
    # We set up a dictionary as such so that we can easily index
    # the values using a string descriptor.
    
    device_attributes_tuples = gpu_device.attributes #.iteritems() 
    device_attributes = {}
    
    for k, v in device_attributes_tuples.items():
        device_attributes[str(k)] = v
    
    num_mp = device_attributes['MultiProcessorCount']
    
    # Cores per multiprocessor is not reported by the GPU!  
    # We must use a lookup table based on compute capability.
    # See the following:
    # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    
    cuda_cores_per_mp = { '50' : 128, '51' : 128, '52' : 128, '60' : 64, '61' : 128, '62' : 128, '86' : 128}[compute_capability]
    
    print ('({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores'.format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))
    
    device_attributes.pop('MultiProcessorCount')
    
    for k in device_attributes.keys():
        print('{}: {}'.format(k, device_attributes[k]))

print("Device query finished !!!")        