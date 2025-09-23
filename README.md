# GEM-pRF  

Welcome to **GEM-pRF** - a standalone, plug-and-play software for population receptive field (pRF) mapping, designed for **large-scale data analysis with high accuracy**.  

To understand the theoretical foundations and details of how the software works, please refer to our paper:  
👉 [Mittal et al (2025), *GEM-pRF: GPU-Empowered Mapping of Population Receptive Fields for Large-Scale fMRI Analysis*](https://www.biorxiv.org/content/10.1101/2025.05.16.654560v1)  

---

## 🚀 Installation  

GEM-pRF is written as a **standalone plug-and-play software**. Once you configure your XML file (see [sample config](https://github.com/siddmittal/GEMpRF/blob/main/gem/configs/analysis_configs/analysis_config.xml)), you can directly run the software with:  

```bash
python run_gem.py pPATH_TO_CONFIG_FILE
