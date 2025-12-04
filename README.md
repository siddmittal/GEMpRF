# GEM-pRF

Welcome to **GEM-pRF**, a standalone, GPU-accelerated tool for population receptive field (pRF) mapping, built for **large-scale fMRI analysis**.

For theory and full method details, see our paper:

👉 *Mittal et al. (2025):*  **GEM-pRF: GPU-Empowered Mapping of Population Receptive Fields for Large-Scale fMRI Analysis**
<https://doi.org/10.1016/j.media.2025.103891>


## Documentation

* Documentation and examples are available at: <https://gemprf.github.io/>

* For a deeper look into the mathematical and computational foundations, the paper above is the best reference.


## Installation

GEM-pRF relies on an NVIDIA GPU and CUDA. Make sure your system has:

* A compatible **NVIDIA GPU**
* A matching **CUDA toolkit**
* A matching **NVCC compiler**

### 1. Install GEM-pRF

```bash
pip install gemprf
```

Latest versions:
<https://pypi.org/project/gemprf/>

### 2. Install CuPy (required)

GEM-pRF depends on CuPy, but CuPy must match your CUDA version — so it is **not installed automatically**.

Install the correct CuPy wheel for your system:

* <https://docs.cupy.dev/en/stable/install.html#installing-cupy>
* or via pip, for example:

```bash
pip install cupy-cuda12x
```



>[!Caution]
Install the CuPy variant that matches *your* CUDA version.

You must install CuPy **before running GEM-pRF**.


## Running GEM-pRF

After installing `gemprf` and a compatible CuPy build, you can run GEM-pRF directly from Python.

### Example

```python
import gemprf as gp

gp.run("path/to/your_config.xml")
```

### Configuration files

GEM-pRF uses XML configuration files to define analysis settings.
See a sample config here:

<https://github.com/siddmittal/GEMpRF_Demo/blob/main/sample_configs/sample_config.xml>


---

## Quick workflow

1. Install GEM-pRF → `pip install gemprf`
2. Install the correct CuPy for your CUDA environment
3. Prepare your XML config file
4. Run:

   ```python
   import gemprf as gp
   gp.run("config.xml")
   ```


