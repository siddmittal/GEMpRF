import json
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the JSON data
# json_result_file = "Y:/data/tests/oprf_test/derivatives/prfanalyze-oprf/analysis-02/sub-002/ses-002/sub-002_ses-002_task-prf_run-01_hemi-BOTH_estimates.json"
json_result_file = "D:/results/fmri/31.10_[with vista hrf, without e-term]_sid-fmri-refined_fit.json"
# json_result_file = "D:/code/sid-git/fmri/local-extracted-datasets/oprf_test/BIDS/derivatives/prfanalyze-oprf/analysis-02/sub-002/ses-003/sub-002_ses-003_task-prf_run-01_hemi-BOTH_estimates.json"

with open(json_result_file, 'r') as json_file:
    data = json.load(json_file)

# Filter out entries with NaN values
# filtered_data = [entry for entry in data if not any(np.isnan(value) for value in [entry['Centerx0'], entry['Centery0'], entry['sigmaMajor']])]
filtered_data = [entry for entry in data 
                 if not (any(np.isnan(value) for value in [entry['Centerx0'], entry['Centery0'], entry['sigmaMajor']]) 
                                                 or entry['R2'] < 0.1)]

# Set the variables
groundtruth_position = (-2, 3)
groundtruth_sigma_x = 1.8
groundtruth_sigma_y = 1.8

# Calculate maximum value between groundtruth_sigma_x and groundtruth_sigma_y
max_sigma = max(groundtruth_sigma_x, groundtruth_sigma_y)

# Create a square-shaped plot
fig, ax = plt.subplots()
ax.set_aspect('equal')  # Ensure the aspect ratio is equal

# Plot groundtruth_position as a blue point
ax.plot(groundtruth_position[0], groundtruth_position[1], 'bo', label='Ground Truth Position')

# Plot an ellipse representing groundtruth_sigma_x and groundtruth_sigma_y as a blue dashed line
ellipse = plt.matplotlib.patches.Ellipse(groundtruth_position, 2 * groundtruth_sigma_x, 2 * groundtruth_sigma_y,
                                        fill=False, linestyle='dashed', linewidth=2, edgecolor='b', label='Ground Truth pRF Sigma')
ax.add_patch(ellipse)

# Compute mean values of Centerx0, Centery0, and sigmaMajor
mean_Centerx0 = np.mean([entry['Centerx0'] for entry in filtered_data])
mean_Centery0 = np.mean([entry['Centery0'] for entry in filtered_data])
mean_sigmaMajor = np.mean([entry['sigmaMajor'] for entry in filtered_data])

# Compute standard deviation in Centerx0 and Centery0
sigma_x = np.std([entry['Centerx0'] for entry in filtered_data])
sigma_y = np.std([entry['Centery0'] for entry in filtered_data])

# Plot Centerx0 and Centery0 as black points
ax.plot(mean_Centerx0, mean_Centery0, 'ko', label='Estimated Mean Position')

# Plot an ellipse using mean values as the center and sigma_x and sigma_y as the spread
ellipse = plt.matplotlib.patches.Ellipse((mean_Centerx0, mean_Centery0), 2 * sigma_x, 2 * sigma_y,
                                        fill=False, linestyle='dashed', linewidth=2, edgecolor='k', label='Estimated Centers Std')
ax.add_patch(ellipse)

# Choose 50 random entries and create gray circles
random_entries = random.sample(filtered_data, 50)
for entry in random_entries:
    center_x = entry['Centerx0']
    center_y = entry['Centery0']
    sigma_major = entry['sigmaMajor']
    circle = plt.Circle((center_x, center_y), sigma_major, color='gray', fill=False)
    ax.add_patch(circle)

# Set x and y ticks based on the maximum sigma value
# ax.set_xlim(-max_sigma, max_sigma)
# ax.set_ylim(-max_sigma, max_sigma)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Add a legend
ax.legend()

# Show the plot
plt.grid()

# plt.savefig("D:/results/fmri/31.10_[with vista hrf, without e-term]_sid-fmri-refined_fit.svg")

plt.show()

print()
