import sys
# sys.path.append('/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/dlinhardt/pythonclass')
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")


from PRFclass import PRF
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# sub = '002'
# ses = '002'


###
basePath="Y:/data/tests/gem-paper-simulated-data/analysis/02/BIDS"
results_base_path = "D:/results/comparison-plots/gem-paper-simulated-data/analysis-02" # 'Y:/data/tests/gem_test/derivatives'
run = '01'
task = 'prf'
hemi='BOTH'


def create_and_save_plot(result_path_seperator, selected_method, selected_method_name, analysis, sub, ses, groundtruth_position, groundtruth_sigma):
    if not selected_method_name == 'fprf':
        selected_method.maskVarExp(.1)
    
    if selected_method_name == 'samsrf':
        selected_method.maskSigma(0.025, s_max=None)

    # RESULT Image
    if selected_method_name == 'samsrf':
        Result_image_path = f"{results_base_path}/{selected_method_name}{result_path_seperator}analysis-{analysis}{result_path_seperator}sub-{sub}{result_path_seperator}ses-{ses}{result_path_seperator}sub-{sub}_ses-{ses}_task-prf_run-01_hemi-BOTH_estimates-NEW.svg"
    else:
        Result_image_path = f"{results_base_path}/prfanalyze-{selected_method_name}{result_path_seperator}analysis-{analysis}{result_path_seperator}sub-{sub}{result_path_seperator}ses-{ses}{result_path_seperator}sub-{sub}_ses-{ses}_task-prf_run-01_hemi-BOTH_estimates.svg"

    # Create a square-shaped plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')  # Ensure the aspect ratio is equal

        # Choose 50 random entries and create gray circles
    # random_entries = random.sample(filtered_data, 50)
    # Choose 50 random indices
    random_indices = random.sample(range(len(selected_method.x)), 50)
    for i in range(50):
        center_x = selected_method.x[random_indices][i]
        center_y = selected_method.y[random_indices][i]
        sigma_major = selected_method.s[random_indices][i]
        circle = plt.Circle((center_x, center_y), sigma_major, color='gray', fill=False)
        ax.add_patch(circle)

    # Plot groundtruth_position as a blue point
    ax.plot(groundtruth_position[0], groundtruth_position[1], 'bo', label='Ground Truth Position')

    # Plot an ellipse representing groundtruth_sigma_x and groundtruth_sigma_y as a blue dashed line
    ellipse = plt.matplotlib.patches.Ellipse(groundtruth_position, 2 * groundtruth_sigma, 2 * groundtruth_sigma,
                                            fill=False, linestyle='dashed', linewidth=2, edgecolor='b', label='Ground Truth pRF Sigma')
    ax.add_patch(ellipse)

    # Compute mean values of Centerx0, Centery0, and sigmaMajor while discarding NaN values
    mean_Centerx0 = np.nanmean(selected_method.x)
    mean_Centery0 = np.nanmean(selected_method.y)
    mean_Sigma0 = np.nanmean(selected_method.s)

    # Compute standard deviation in Centerx0 and Centery0 while discarding NaN values
    center_x_std = np.nanstd(selected_method.x)
    center_y_std = np.nanstd(selected_method.y)

    # Plot Centerx0 and Centery0 as black points
    ax.plot(mean_Centerx0, mean_Centery0, 'ko', label='Estimated Mean Position')

    # Plot an ellipse using mean values as the center and sigma_x and sigma_y as the spread
    ellipse = plt.matplotlib.patches.Ellipse((mean_Centerx0, mean_Centery0), 2 * mean_Sigma0, 2 * mean_Sigma0,
                                            fill=False, linestyle='dashed', linewidth=2, edgecolor='k', label='Estimated Centers Std')
    ax.add_patch(ellipse)



    # Set x and y ticks based on the maximum sigma value
    ax.set_xlim(groundtruth_position[0] - 2*groundtruth_sigma, groundtruth_position[0] + 2*groundtruth_sigma)
    ax.set_ylim(groundtruth_position[1]-2*groundtruth_sigma, groundtruth_position[1] + 2*groundtruth_sigma)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add a legend
    # ax.legend()

    # Show the plot
    plt.grid()

    plt.savefig(Result_image_path, bbox_inches='tight')

    # plt.show()

    
    print()


def fprf_create_and_save_plot(result_path_seperator, selected_method, selected_method_name, analysis, sub, ses, groundtruth_position, groundtruth_sigma):
    if not selected_method_name == 'fprf':
        selected_method.maskVarExp(.1)

    # RESULT Image
    if selected_method_name == 'samsrf':
        Result_image_path = f"{results_base_path}/{selected_method_name}{result_path_seperator}analysis-{analysis}{result_path_seperator}sub-{sub}{result_path_seperator}ses-{ses}{result_path_seperator}sub-{sub}_ses-{ses}_task-prf_run-01_hemi-BOTH_estimates.svg"
    else:
        Result_image_path = f"{results_base_path}/prfanalyze-{selected_method_name}{result_path_seperator}analysis-{analysis}{result_path_seperator}sub-{sub}{result_path_seperator}ses-{ses}{result_path_seperator}sub-{sub}_ses-{ses}_task-prf_run-01_hemi-BOTH_estimates.svg"

    # Create a square-shaped plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')  # Ensure the aspect ratio is equal

        # Choose 50 random entries and create gray circles
    # random_entries = random.sample(filtered_data, 50)
    # Choose 50 random indices
    random_indices = random.sample(range(len(selected_method.x0)), 50)
    for i in range(50):
        center_x = selected_method.x0[random_indices][i]
        center_y = selected_method.y0[random_indices][i]
        sigma_major = selected_method.s0[random_indices][i] / 5
        circle = plt.Circle((center_x, center_y), sigma_major, color='gray', fill=False)
        ax.add_patch(circle)

    # Plot groundtruth_position as a blue point
    ax.plot(groundtruth_position[0], groundtruth_position[1], 'bo', label='Ground Truth Position')

    # Plot an ellipse representing groundtruth_sigma_x and groundtruth_sigma_y as a blue dashed line
    ellipse = plt.matplotlib.patches.Ellipse(groundtruth_position, 2 * groundtruth_sigma, 2 * groundtruth_sigma,
                                            fill=False, linestyle='dashed', linewidth=2, edgecolor='b', label='Ground Truth pRF Sigma')
    ax.add_patch(ellipse)

    # Compute mean values of Centerx0, Centery0, and sigmaMajor while discarding NaN values
    mean_Centerx0 = np.nanmean(selected_method.x0)
    mean_Centery0 = np.nanmean(selected_method.y0)
    mean_Sigma0 = np.nanmean(selected_method.s) / 5

    # Compute standard deviation in Centerx0 and Centery0 while discarding NaN values
    center_x_std = np.nanstd(selected_method.x0)
    center_y_std = np.nanstd(selected_method.y0)

    # Plot Centerx0 and Centery0 as black points
    ax.plot(mean_Centerx0, mean_Centery0, 'ko', label='Estimated Mean Position')

    # Plot an ellipse using mean values as the center and sigma_x and sigma_y as the spread
    ellipse = plt.matplotlib.patches.Ellipse((mean_Centerx0, mean_Centery0), 2 * mean_Sigma0, 2 * mean_Sigma0,
                                            fill=False, linestyle='dashed', linewidth=2, edgecolor='k', label='Estimated Centers Std')
    ax.add_patch(ellipse)



    # Set x and y ticks based on the maximum sigma value
    ax.set_xlim(groundtruth_position[0] - 2*groundtruth_sigma, groundtruth_position[0] + 2*groundtruth_sigma)
    ax.set_ylim(groundtruth_position[1]-2*groundtruth_sigma, groundtruth_position[1] + 2*groundtruth_sigma)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add a legend
    # ax.legend()

    # Show the plot
    plt.grid()

    plt.savefig(Result_image_path, bbox_inches='tight')

    # plt.show()
    print()


##############################-----------------------------MAIN()---------------------------------#####################
analysis_vista = '01' #'02'
analysis_gem = '02'
analysis_fprf = '03' # with same HRF # '02' 
analysis_samsrf = '04' # '02'

groundtruth_positions = [(0, 0), (3, 3), (6, 6), (9, 9)]
groundtruth_sigmas = [1, 1, 1, 1]

subjects = ['001', '002', '003', '004']
sessions = ['001', '002', '003']

# groundtruth_positions = [(6, 6)]
# groundtruth_sigmas = [1, 1, 1, 1]

# subjects = ['003']
# sessions = ['001', '002', '003']

slash_or_hyphen = '_'

counter = 0
for sub in subjects:
    for ses in sessions:    
        # Methods data
        # vista = PRF.from_docker('', sub, ses, task, '01', analysis=analysis_vista, hemi=hemi, baseP=basePath, orientation='MP', method='vista')

        estimate_results_path = f"Y:/data/tests/gem-paper-simulated-data/analysis/02/BIDS/derivatives/prfanalyze-gem/analysis-02/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-prf_acq-normal_run-01_estimates.json"
        # gem = PRF.from_file('', sub, ses, task, '01', analysis=analysis_gem, hemi=hemi, baseP=basePath, orientation='MP', method='gem') # <----Changed to 'MP'
        gem = PRF.from_file("gem-paper-simulated-data", estimate_results_path, "MP") # <----Changed to 'MP'

        
        # fprf = PRF.from_docker('', sub, ses, task, '01', analysis=analysis_fprf, hemi=hemi, baseP=basePath, orientation='MP', method='fprf')
        # samsrf = PRF.from_samsrf('', sub, ses, task, '01', analysis=analysis_samsrf,baseP=basePath, orientation='MP')

        # create_and_save_plot(result_path_seperator= slash_or_hyphen, selected_method=vista, selected_method_name='vista', analysis = analysis_vista, sub=sub, ses=ses, groundtruth_position=groundtruth_positions[counter], groundtruth_sigma=groundtruth_sigmas[counter])
        create_and_save_plot(result_path_seperator= slash_or_hyphen, selected_method=gem, selected_method_name='gem', analysis = analysis_gem, sub=sub, ses=ses, groundtruth_position=groundtruth_positions[counter], groundtruth_sigma=groundtruth_sigmas[counter])        
        # fprf_create_and_save_plot(result_path_seperator= slash_or_hyphen, selected_method=fprf, selected_method_name='fprf', analysis = analysis_fprf, sub=sub, ses=ses, groundtruth_position=groundtruth_positions[counter], groundtruth_sigma=groundtruth_sigmas[counter])
        # create_and_save_plot(result_path_seperator= slash_or_hyphen, selected_method=samsrf, selected_method_name='samsrf', analysis = analysis_samsrf, sub=sub, ses=ses, groundtruth_position=groundtruth_positions[counter], groundtruth_sigma=groundtruth_sigmas[counter])    
        print('Done plotting....')
    counter = counter + 1

print("Finsihed program...")    