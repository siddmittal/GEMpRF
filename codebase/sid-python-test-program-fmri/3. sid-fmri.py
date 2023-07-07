import nibabel as nib
from nilearn import plotting

# Load the fMRI image
fmri_img = nib.load('sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz')

# Extract the first volume (3D image) from the 4D data
fmri_data = fmri_img.get_fdata()
volume = fmri_data[..., 0]

# Convert the volume to a nibabel-compatible format
nib_volume = nib.Nifti1Image(volume, affine=fmri_img.affine)

# Plot the transversal slice images
plotting.plot_epi(nib_volume, display_mode='z', cut_coords=10, title="Transversal Slice Images")

# Show the plots
plotting.show()
