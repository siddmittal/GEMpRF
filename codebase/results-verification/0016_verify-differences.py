import json
import numpy as np

# Loading JSON data
with open("Z:/home/smittal/multi-gpu/results/2024-01-10_sub-sidtest_ses-001_task-bar_run-01_hemi-L_estimates_[multi-gpu_multi-batching-numba-arr-v-5_51x51x8].json", 'r') as method1_file:
    gem_data = json.load(method1_file)

with open("Z:/home/smittal/multi-gpu/results/2024-01-07_sub-sidtest_ses-001_task-bar_run-01_hemi-L_estimates_[multi-gpu].json", 'r') as method2_file:
    oprf_data = json.load(method2_file)

# Replace NaN with a very high value
high_value = 1e9  # You can adjust this value based on your data

# Handling NaN values in gem_data
for entry in gem_data:
    for k, v in entry.items():
        if isinstance(v, list):
            entry[k] = [high_value if (x is None or np.isnan(x)) else x for x in v]
        else:
            entry[k] = high_value if (v is None or np.isnan(v)) else v

# Handling NaN values in oprf_data
for entry in oprf_data:
    for k, v in entry.items():
        if isinstance(v, list):
            entry[k] = [high_value if (x is None or np.isnan(x)) else x for x in v]
        else:
            entry[k] = high_value if (v is None or np.isnan(v)) else v

# Extracting data
gem_centerx0 = [entry['Centerx0'] for entry in gem_data]
gem_centery0 = [entry['Centery0'] for entry in gem_data]
gem_sigma = [entry['sigmaMajor'] for entry in gem_data]
gem_r2 = [entry['R2'] for entry in gem_data]

oprf_centerx0 = [entry['Centerx0'] for entry in oprf_data]
oprf_centery0 = [entry['Centery0'] for entry in oprf_data]
oprf_sigma = [entry['sigmaMajor'] for entry in oprf_data]
oprf_r2 = [entry['R2'] for entry in oprf_data]

# Calculating differences
max_difference_x0 = np.max(np.array(gem_centerx0) - np.array(oprf_centerx0))
max_difference_y0 = np.max(np.array(gem_centery0) - np.array(oprf_centery0))
max_difference_s0 = np.max(np.array(gem_sigma) - np.array(oprf_sigma))
max_difference_r2 = np.max(np.array(gem_r2) - np.array(oprf_r2))

print(f"Differences: x0:{max_difference_x0}, y0:{max_difference_y0}, sigma:{max_difference_s0}, r2:{max_difference_r2}")
print()
