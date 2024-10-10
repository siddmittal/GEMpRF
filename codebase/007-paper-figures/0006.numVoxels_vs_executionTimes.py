import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_filepath = r"/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/05/BIDS/derivatives/time_records/MergedTimeResults.xlsx"

# Read the data from the .xlsx file into a Pandas DataFrame
data = pd.read_excel(input_filepath)

# Create a plot using Seaborn
plt.figure(figsize=(10, 6))


# DGX system results 
sns.lineplot(x="NumVoxels", y="Time_seconds_dgx_51x51x8", data=data, marker='o', label='NVIDIA DGX V100, Grid Size: 51x51x8')
sns.lineplot(x="NumVoxels", y="Time_seconds_dgx_151x151x16", data=data, marker='o', label='NVIDIA DGX V100, Grid Size: 151x151x16')

# ASUS results
sns.lineplot(x="NumVoxels", y="Time_seconds_asus_51x51x8", data=data, marker='o', label='NVIDIA GeForce RTX 3050, Grid Size: 51x51x8')

# plot
plt.xlabel('NumVoxels')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. NumVoxels')
plt.legend()
plt.grid(True)
plt.show()