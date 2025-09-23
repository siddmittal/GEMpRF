Welcome to **GEM-pRF** - a standalone, plug-and-play software for population receptive field (pRF) mapping, designed for **large-scale data analysis with high accuracy**.


To understand the theoretical foundations and details of how the software works, please refer to our paper: 👉[Mittal et al (2025), ](https://www.biorxiv.org/content/10.1101/2025.05.16.654560v1)*[GEM-pRF: GPU-Empowered Mapping of Population Receptive Fields for Large-Scale fMRI Analysis](https://www.biorxiv.org/content/10.1101/2025.05.16.654560v1)*


## Documentation

An official documentation is coming soon ([GEM-pRF documentation link](https://gemprf.github.io/))! Meanwhile, to get the mathematical foundation of the software, you may refer to the [GEM-pRF paper](https://www.biorxiv.org/content/10.1101/2025.05.16.654560v1).


## Installation

GEM-pRF requires the GPU access for the data processing. At the moment, GEM uses CUDA libraries to acess/process data on NVIDIA GPUs.

> \[!WARNING\]
>
> Please check your system has compatible NVIDIA GPU available.

### Step-by-Step Guide

**Step 1. Install dependencies**

* Create or activate your preferred Python/Conda environment.
* Install all required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```


**Step 2. Download GEM-pRF code**

* Clone the repository:

```bash
git clone https://github.com/siddmittal/GEMpRF.git
cd GEMpRF
```


## Running GEM-pRF


> \[!CAUTION\]
> Before proceeding, make sure to install the required python dependencies as specified in the `requirements.txt` file

GEM-pRF is written as a **standalone software**. It comes with an XML configuration file. Once you configure your XML file (see [sample config](https://github.com/siddmittal/GEMpRF/blob/main/gem/configs/analysis_configs/analysis_config.xml)), you can directly run the software.

### 🔹 **Option A: Run from terminal**




1. Open a terminal (e.g. Anaconda Prompt).
2. Activate the environment with the dependencies installed.
3. Navigate to the GEM-pRF folder.
4. Run:

   ```bash
   python run_gem.py PATH_TO_YOUR_XML_CONFIG_FILE
   ```


### 🔹 **Option B: Run from IDE (e.g. VS Code)**




1. Open the downloaded GEM-pRF folder in VS Code.
2. Edit the `run_gem.py` script to specify the path to your XML config file.
3. Run the script directly from the IDE.


