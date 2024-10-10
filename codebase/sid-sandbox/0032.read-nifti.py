import nibabel as nib
import numpy as np

# Load the NIfTI file
nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-vista/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_modelpred.nii.gz"
img = nib.load(nifti_file)

# Get the NIfTI data as a NumPy array
data = img.get_fdata()

nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_centerx0.nii.gz"
img = nib.load(nifti_file)

# Get the NIfTI data as a NumPy array
data2 = img.get_fdata()



# Modify the data as needed
# For example, let's set all values to 1
new_data = np.ones(data.shape)

# Create a new NIfTI image with the modified data
new_img = nib.Nifti1Image(new_data, img.affine)

# Save the NIfTI data with the modified data back to a new NIfTI file
output_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_centerx0.nii.gz"
nib.save(new_img, output_file)


# import nibabel as nib
# import numpy as np

# def save_1d_numpy_as_nifti(data, output_file):
#     # Ensure the input data is a 1D NumPy array
#     if data.ndim != 1:
#         raise ValueError("Input data must be a 1D NumPy array.")
    
#     # Reshape the data to the specified shape (2, 1, 1, 1)
#     data = data.reshape((2, 1, 1, 1))

#     # Create a NIfTI image with the reshaped data
#     nifti_image = nib.Nifti1Image(data, np.eye(4))  # Identity affine matrix

#     # Save the NIfTI image to the specified output file
#     nib.save(nifti_image, output_file)

# # Usage example:
# data = np.array([0.25701308, 0.04549263])
# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_centerx0.nii.gz"
# save_1d_numpy_as_nifti(data, output_file)

# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_centerx0.nii.gz"
# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_centery0.nii.gz"
# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_theta.nii.gz"
# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_sigmaminor.nii.gz"
# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_sigmamajor.nii.gz"
# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_r2.nii.gz"
# output_file = nifti_file = "D:/results/docker-analysis/01-test/BIDS/derivatives/prfanalyze-oprf/sub-001/ses-20200320/sub-001_ses-20200320_task-prf_acq-normal_run-01_modelpred.nii.gz"

