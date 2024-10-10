## sub-001, ses-005, run-04, voxel-27
## sub-002, ses-006, run-03
## sub-001, ses-006, run-04, voxel-90

import sys
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")
from PRFclass import PRF

import os
import json
import numpy as np

# for plotting
import matplotlib.pyplot as plt

R2_vs_LINESLOPE_PLOT_DRAWN = False

subs = ['001']
# sessions = ['001', '002']
# runs = ['01', '02', '03']

# subs = ['001', '002']
sessions = ['001', '002', '003', '004', '005', '006']
runs = ['01', '02', '03', '04', '05', '0102030405avg']

hemis = ['L'] #['L', 'R']

fmri_measured_basepath = f"D:/results/with-without-nordic-covmap/prfprepare/analysis-01"
fmri_measured_signal_length = None

def load_all_data(pRF_estimations_basepath, isStandard : bool = False):    
    hemi = 'L'
    
    counter = 0
    subject_data = {}
    for sub in subs:
        sessions_data = {}
        for ses in sessions:
            runs_data = {}
            for run in runs:
                if isStandard:
                    pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                else:
                    pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_estimates.json"

                if os.path.exists(pRF_estimations_filepath):
                    with open(pRF_estimations_filepath, 'r') as file:
                        single_run_json_data = json.load(file)
                        # print(f"{counter}: pRF Size at Voxel 90: {single_run_json_data[90]['sigmaMajor']}")
                        counter += 1

                runs_data[run] = single_run_json_data
            sessions_data[ses] = runs_data
        subject_data[sub] = sessions_data

    return subject_data

###################################------------------------------------------------Data Loading-------------------------------------------------############################################
# Load all data into a DataFrame
gem_pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/analysis-03_AsusCorrect"    
vista_pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/prfanalyze-vista/analysis-01"
#...gem data
all_data_standard_gem = load_all_data(pRF_estimations_basepath=gem_pRF_estimations_basepath, isStandard=True)

#...vista data
all_data_standard_vista = load_all_data(pRF_estimations_basepath=vista_pRF_estimations_basepath, isStandard=True)

#...gem-nordic data
all_data_nordic_gem = load_all_data(pRF_estimations_basepath=gem_pRF_estimations_basepath, isStandard=False)


# GET the pRF Size values for a given voxel
def extract_prf_size_values(selected_method_data, voxel_number):    
    # GEM voxel's data
    prf_size_values = []
    for sub in subs:
        for ses in sessions:
            for run in runs:
                run_data = selected_method_data[sub][ses][run]
                prf_size_values.append(run_data[voxel_number]['sigmaMajor'])

    return prf_size_values

def get_all_methods_pRF_sizes_values(voxel_number):
    global all_data_standard_gem, all_data_nordic_gem, all_data_standard_vista
    gem_standard_prf_size_values = extract_prf_size_values(all_data_standard_gem, voxel_number)
    gem_nordic_prf_size_values = extract_prf_size_values(all_data_nordic_gem, voxel_number)
    vista_standard_prf_size_values = extract_prf_size_values(all_data_standard_vista, voxel_number)

    return gem_standard_prf_size_values, vista_standard_prf_size_values, gem_nordic_prf_size_values

def plot_pRF_size_distributon(voxel_number):
    gem_points, vista_points, nordic_points = get_all_methods_pRF_sizes_values(voxel_number)
    run_indices = np.arange(1, len(gem_points)+1)

    fig, ax = plt.subplots()
    ax.scatter(run_indices, gem_points, color='blue', label='Standard-Gem')
    ax.scatter(run_indices, vista_points, color='orange', label='Standard-Vista')
    ax.scatter(run_indices, nordic_points, color='green', label='Nordic-Gem')
    # add the legends
    ax.legend()
    # add the labels
    ax.set_xlabel('Run Index')
    ax.set_ylabel('pRF Size')
    ax.set_title(f'pRF Size values for voxel {voxel_number}')

    # legends to specify the color of each method
    plt.legend(loc='upper right', fontsize=10)
    
    plt.show()


# plot the pRF Size values for a given voxel
voxel_number = 90
plot_pRF_size_distributon(voxel_number)



