import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import math
import nibabel as nib

# nifti data
# y_signals_path = r"Y:\data\stimsim23\derivatives\prfprepare\analysis-01\sub-sidtest\ses-002\func\sub-sidtest_ses-002_task-bar_run-02_hemi-R_bold.nii.gz"
y_signals_path = r"D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data\sub-sidtest_ses-001_task-bar_run-01_hemi-L_bold.nii.gz"
bold_response_img = nib.load(y_signals_path)
Y_signals_cpu = bold_response_img.get_fdata()
Y_signals_cpu = (Y_signals_cpu.reshape(-1, Y_signals_cpu.shape[-1])).T
row_means = Y_signals_cpu.mean(axis=1)
Y_signals_cpu -= row_means.reshape(-1, 1) # substract mean
Y_signals_cpu /= Y_signals_cpu.max(axis=1).reshape(-1, 1)

# sigma_test = np.linspace(float(0.5), float(5), int(8)) # 0.5 to 1.5
# errors = np.array([[1, 3, 2, 1], [1, 3, 4, 5], [2, 3, 1, 2], [3, 2, 1, 0]])
# best = np.argmax(errors, axis=1)
# best_2 = np.argmax(errors[:, 1:-1], axis=1)


# df3 = pd.read_excel("D:/results/gradients-test/29.01.2024-refinement-[quad-vs-line-SIMULATIONS]_fx-51x51x8.xlsx", index_col=None)
# df3 = pd.read_excel("D:/results/gradients-test/29.01.2024-refinement-[quad-vs-line-EMPIRICAL]_fx-51x51x8.xlsx", index_col=None)
# df3 = pd.read_excel(r"D:\results\gradients-test\2024-02-05_gradients-analysis_[signals_simga-14c_boundary-1_quad-vs-line-EMPIRICAL]_fx-51x51x8.xlsx", index_col=None)
#df3 = pd.read_excel(r"D:\results\gradients-test\2024-02-05_gradients-analysis_[signals_simga-14c_boundary-1_quad-vs-line-EMPIRICAL-largeData]_fx-51x51x8.xlsx", index_col=None)
# df3 = pd.read_excel(r"D:\results\gradients-test\2024-02-09_gradients-analysis_[fex_error+signals_simga-14c_boundary-1_quad-vs-line-EMPIRICAL-smallData_run-01_hemi-L]_fx-101x101x16.xlsx", index_col=None)
# df3 = pd.read_excel(r"D:\results\gradients-test\2024-02-09_gradients-analysis_[signals_simga-14c_boundary-1_quad-vs-line-EMPIRICAL-smallData_run-01_hemi-L]_fx-51x51x8.xlsx", index_col=None)
# df3 = pd.read_excel(r"D:\results\gradients-test\2024-02-14_gradients-analysis_[weight-real_fex_error+signals_simga-14c_boundary-1_quad-vs-line-EMPIRICAL-smallData_run-01_hemi-L]_fx-51x51x8.xlsx", index_col=None)
# df3 = pd.read_excel(r"D:\results\gradients-test\2024-02-15_gradients-analysis_[weight-real_fex_error+signals_simga-14c_boundary-1_quad-vs-line-EMPIRICAL-smallData_run-01_hemi-L]_fx-51x51x8.xlsx", index_col=None)
df3 = pd.read_excel(r"D:\results\gradients-test\2024-02-16_gradients-analysis_[error-check_v2_51x51x8].xlsx", index_col=None)

coarse_est = df3['CoarseEstimations']
quad_refine_est = df3['QuadRefinedEstimations']
coarse_errors = df3['CoarseErrors']
quad_errors = df3['QuadErrors']
quad_r2 = df3['QuadR2'].to_numpy()
model_signals_strings = df3['MatchedModelSignal']
y_signals_strings = df3['YSignal']
# fex_error = np.array(df3['FexError'])

coarse_result = []
quad_refine_result = []
coarse_errors_result = []
quad_errors_result = []
model_signals = []
y_signals = []

for idx in range(len(coarse_est)):
    # estimaitons
    coarse_est_values = ast.literal_eval(coarse_est[idx])
    quad_refine_est_values = [float(val) for val in quad_refine_est[idx][1:-1].split()] #ast.literal_eval(refine_est[idx])

    # errors
    # coarse_error_values = np.array([0, 0, 0, 0]) if (coarse_errors[idx]).find('nan') !=-1  else ast.literal_eval(coarse_errors[idx])
    # quad_error_values = np.array([0, 0, 0, 0]) if (quad_errors[idx]).find('nan') !=-1  else ast.literal_eval(quad_errors[idx])
    coarse_error_values = np.array([np.nan, np.nan, np.nan, np.nan]) if (coarse_errors[idx]).find('nan') !=-1  else ast.literal_eval(coarse_errors[idx])
    quad_error_values = np.array([np.nan, np.nan, np.nan, np.nan]) if (quad_errors[idx]).find('nan') !=-1  else ast.literal_eval(quad_errors[idx])

    # signals
    if (isinstance(model_signals_strings[idx], str)):
        m_s = np.array([float(val) for val in model_signals_strings[idx][1:-1].replace('\n', '').split()])
        y_s = np.array([float(val) for val in y_signals_strings[idx][1:-1].replace('\n', '').split()])
    else:
        m_s = np.zeros((300))
        y_s = np.zeros((300))

    # append values
    coarse_result.append(np.array(coarse_est_values))
    quad_refine_result.append(np.array(quad_refine_est_values))

    coarse_errors_result.append(np.array(coarse_error_values))
    quad_errors_result.append(np.array(quad_error_values))
    model_signals.append(m_s)
    y_signals.append(y_s)

coarse_muX = (np.array(coarse_result))[:, 0]
coarse_muY = (np.array(coarse_result))[:, 1]
coarse_sigma = (np.array(coarse_result))[:, 2]

quad_sigma = (np.array(quad_refine_result))[:, 2]

coarse_errors = (np.array(coarse_errors_result))[:, 0]
quad_errors = (np.array(quad_errors_result))[:, 0]

# plots
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Subplot 1: Individual arrays
ax1.plot(coarse_errors, label='Coarse Errors')
ax1.plot(quad_errors, label='Quad Errors')
ax1.set_ylabel('Errors')
ax1.legend()

# Subplot 2: Differences among arrays
ax2.plot(coarse_errors - quad_errors, label='Coarse - Quad', color='red')
ax2.set_xlabel('Data Points')
ax2.set_ylabel('Differences')
ax2.legend()

# Adjust layout to prevent clipping of legends
plt.tight_layout()
 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

# Subplot 2: Differences among arrays
ax1.hist(coarse_errors - quad_errors, 60, label='Coarse - Quad', color='red')

# Adjust layout to prevent clipping of legends
plt.tight_layout()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.scatter(coarse_sigma, coarse_errors - quad_errors)

plt.figure()
error_mask = ((coarse_errors - quad_errors) > 0)
mask = (coarse_sigma < 0.7) | (coarse_sigma > 4.0)
r2_mask = (quad_r2 < 1) & (quad_r2 > -3)
plt.plot(coarse_sigma[mask])
plt.plot(quad_sigma[mask])
plt.plot(quad_r2[mask & r2_mask])


# indices for worsen errors, high r2
worse_errors_high_r2 = ((coarse_errors - quad_errors) > 0) & (quad_r2 > 0.15) # Ideally coarse errors should be small as comapred to refined errors
# worse_errors_high_r2 = np.ones(coarse_errors.shape, dtype=bool)#((coarse_errors - quad_errors) < 0) & (quad_r2 > 0)
worse_error_coarse_gradients = ((np.array(coarse_errors_result))[:, 1:4])[worse_errors_high_r2]
worse_error_quad_gradients = ((np.array(quad_errors_result))[:, 1:4])[worse_errors_high_r2]

# worse_error_line_gradients = ((np.array(line_errors_result))[:, 1:4])[worse_errors_high_r2]

# indices for low errors, high r2
good_errors_high_r2 = ((coarse_errors - quad_errors) < 0) & (quad_r2 > 0.6)
good_error_coarse_gradients = ((np.array(coarse_errors_result))[:, 1:4])[good_errors_high_r2]
good_error_quad_gradients = ((np.array(quad_errors_result))[:, 1:4])[good_errors_high_r2]

# Error vs Gradient plot
plt.figure()
plt.subplot(3, 1, 1)
plt.scatter((coarse_errors - quad_errors)[worse_errors_high_r2], worse_error_coarse_gradients[:, 0])
plt.title('Error vs de/dx')
plt.subplot(3, 1, 2)
plt.scatter((coarse_errors - quad_errors)[worse_errors_high_r2], worse_error_coarse_gradients[:, 1])
plt.title('Error vs de/dy')
plt.subplot(3, 1, 3)
plt.scatter((coarse_errors - quad_errors)[worse_errors_high_r2], worse_error_coarse_gradients[:, 2])
plt.title('Error vs de/dsigma')
plt.tight_layout()  

# Debugging low error worsening
# low_error_worsening_idx = np.array([3483, 5176, 6512, 13658, 14059])
# low_error_worsening_params = np.column_stack((coarse_muX[low_error_worsening_idx], coarse_muY[low_error_worsening_idx], coarse_sigma[low_error_worsening_idx]))


# NOTE: some of the values in "quad_errors" are actaully NaN therefore, "sub_results" may show very high error which in reality is just the signals with NaN estimations
sub_results = (coarse_errors - quad_errors)
low_error_high_r2_values = sub_results[(sub_results>0) & (sub_results<1) & (quad_r2 > 0.15)]
np.argwhere((sub_results>0) & (sub_results<1) & (quad_r2 > 0.15))

# max value of worse error - We now have some numerical errors. Even though we are replacing the worsening refined estimations with coarse estimations, still the errors are not exactly same
all_worse_errors = sub_results[(sub_results>0) & (~np.isnan(sub_results>0))]
print(f'Max worse error: {np.max(all_worse_errors)}')
print(f'sub_results[np.argwhere((sub_results>0) & (sub_results<1) & (quad_r2 > 0.15))] {sub_results[np.argwhere((sub_results>0) & (sub_results<1) & (quad_r2 > 0.15))]}')

# fex error
# sub_results_fex = (coarse_errors - fex_error)
# np.argwhere((sub_results_fex>0) & (sub_results_fex<1) & (quad_r2 > 0.15))


# Display the plots
plt.show() 




print