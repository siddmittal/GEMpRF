import numpy as np
import cupy as cp

# singleton
class SearchSpace:
    _instance = None

    @staticmethod
    def get_instance():
        if SearchSpace._instance is None:
            raise Exception("SearchSpace has not been initialized. Call initialize_instance() first.")
        return SearchSpace._instance

    @staticmethod
    def initialize_instance(search_space_xx, search_space_yy, search_space_sigma_range):
        if SearchSpace._instance is None:
            SearchSpace._instance = SearchSpace()

            # CPU variables
            SearchSpace._instance._search_space_xx_cpu = search_space_xx
            SearchSpace._instance._search_space_yy_cpu = search_space_yy
            SearchSpace._instance._search_space_sigma_range_cpu = search_space_sigma_range
            SearchSpace._instance._meshgrid_X, SearchSpace._instance._meshgrid_Y, SearchSpace._instance._meshgrid_Sigma = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range)

            # GPU variables
            SearchSpace._instance._search_space_xx_gpu = cp.asarray(search_space_xx)
            SearchSpace._instance._search_space_yy_gpu = cp.asarray(search_space_yy)
            SearchSpace._instance._search_space_sigma_range_gpu = cp.asarray(search_space_sigma_range)

        return SearchSpace._instance

    @property
    def search_space_xx_cpu(self):
        return self._instance._search_space_xx_cpu

    @property
    def search_space_yy_cpu(self):
        return self._instance._search_space_yy_cpu

    @property
    def search_space_sigma_range_cpu(self):
        return self._instance._search_space_sigma_range_cpu

    @property
    def meshgrid_X(self):
        return self._instance._meshgrid_X

    @property
    def meshgrid_Y(self):
        return self._instance._meshgrid_Y

    @property
    def meshgrid_Sigma(self):
        return self._instance._meshgrid_Sigma

    @property
    def search_space_xx_gpu(self):
        return self._instance._search_space_xx_gpu

    @property
    def search_space_yy_gpu(self):
        return self._instance._search_space_yy_gpu

    @property
    def search_space_sigma_range_gpu(self):
        return self._instance._search_space_sigma_range_gpu
