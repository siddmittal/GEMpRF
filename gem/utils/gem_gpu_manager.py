# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Siddharth Mittal",
"@Desc    :   This class is used to set the default GPU and use the instance of this class to call different methods provided by CuPy library",     
"""
import cupy as cp

class GemGpuManager:
    _instance = None  # Private class variable to store the singleton instance

    def __new__(cls, default_gpu_id):
        if cls._instance is None:
            cls._instance = super(GemGpuManager, cls).__new__(cls)
            cls._instance.__initialize(default_gpu_id)
        return cls._instance

    def __initialize(self, default_gpu_id):
        """Initialize method to set the default GPU ID."""
        self.__default_gpu_id = default_gpu_id
        self.__gem_default_device = cp.cuda.Device(default_gpu_id)

    @classmethod
    def get_instance(cls):
        """Class method to get the singleton instance."""
        return cls._instance

    @property
    def default_gpu_id(self):
        """
        Returns the default GPU ID.
        Returns:
            int: The ID of the default GPU.
        """

        return self.__default_gpu_id

    def execute_cupy_func(self, cupy_func, execute_on_custom_default_gpu=False, cupy_func_args=(), cupy_func_kwargs={}):
        """
        Executes a CuPy function with the provided arguments and keyword arguments.
        Parameters:
        -----------
        cupy_func : callable
            The CuPy function to be executed.
        execute_on_custom_default_gpu : bool, optional
            If True, the function will be executed on the custom default GPU specified by `self.default_gpu_id`.
            Default is False.
        cupy_func_args : tuple, optional
            The positional arguments to be passed to the CuPy function. Default is an empty tuple.
        cupy_func_kwargs : dict, optional
            The keyword arguments to be passed to the CuPy function. Default is an empty dictionary.
        Returns:
        --------
        result
            The result of the CuPy function execution.
        Raises:
        -------
        TypeError
            If `cupy_func_args` is not a tuple or `cupy_func_kwargs` is not a dictionary.
        """

        if not isinstance(cupy_func_args, tuple):
            raise TypeError("cupy function args must be a tuple")
        if not isinstance(cupy_func_kwargs, dict):
            raise TypeError("cupy function kwargs must be a dictionary")

        if execute_on_custom_default_gpu:
            with cp.cuda.Device(self.default_gpu_id):
                return cupy_func(*cupy_func_args, **cupy_func_kwargs)
        else:
            return cupy_func(*cupy_func_args, **cupy_func_kwargs)
        
    def execute_cupy_func_on_default(self, cupy_func, cupy_func_args=(), cupy_func_kwargs={}):
        return self.execute_cupy_func(cupy_func, execute_on_custom_default_gpu=True, cupy_func_args=cupy_func_args, cupy_func_kwargs=cupy_func_kwargs)

# Run main
if __name__ == "__main__":
    import numpy as np
    # test arrays
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    custom_default_gpu = 1
    ggm = GemGpuManager(default_gpu_id=custom_default_gpu)
    print(ggm.default_gpu_id)

    a_d = ggm.execute_cupy_func(cp.asarray, execute_on_custom_default_gpu = True, cupy_func_args=(a,))
    b_d = ggm.execute_cupy_func(cp.asarray, execute_on_custom_default_gpu = True, cupy_func_args=(b,))
    c_d = ggm.execute_cupy_func(cp.add, execute_on_custom_default_gpu = True, cupy_func_args=(a_d, b_d))
    print(f"a_d is on device: {a_d.device.id}")
    print(f"b_d is on device: {b_d.device.id}")
    print(f"c_d is on device: {c_d.device.id}")
    print(f"Value of c_d: {c_d}")

    # use another default device
    ggm = GemGpuManager.get_instance()
    print(ggm.default_gpu_id)
    a_d = ggm.execute_cupy_func(cp.asarray, execute_on_custom_default_gpu = True, cupy_func_args=(a,))
    b_d = ggm.execute_cupy_func(cp.asarray, execute_on_custom_default_gpu = True, cupy_func_args=(b,))
    c_d = ggm.execute_cupy_func(cp.add, execute_on_custom_default_gpu = True, cupy_func_args=(a_d, b_d))
    print(f"a_d is on device: {a_d.device.id}")
    print(f"b_d is on device: {b_d.device.id}")
    print(f"c_d is on device: {c_d.device.id}")
    print(f"Value of c_d: {c_d}")