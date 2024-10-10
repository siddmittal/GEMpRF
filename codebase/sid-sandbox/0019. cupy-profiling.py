# import cupy as cp
# import cProfile


# def my_gpu_function():
#     # CuPy GPU operations
#     data = cp.random.rand(1000, 1000)
#     result = cp.dot(data, data)

# if __name__ == '__main__':
#     cProfile.run('my_gpu_function()', sort='cumulative')



import cupy as cp

# CuPy GPU operations
data = cp.random.rand(1000, 1000)
result = cp.dot(data, data)

