import os
import matplotlib.pyplot as plt
from nilearn import image as nimg
from nilearn import plotting as nplot

file_paths = [
    "sub-sidtest_ses-001_task-bar_run-01_hemi-L_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-01_hemi-R_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-02_hemi-L_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-02_hemi-R_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-0102avg_hemi-R_bold.nii.gz",
    "task-bar_apertures.nii.gz"
]

t1 = file_paths[6]

t1_img_4d = nimg.load_img(t1)

# Select the middle slice from the third dimension (z axis)
z_slice = t1_img_4d.shape[2] // 2
t1_img_3d = nimg.index_img(t1_img_4d, z_slice)

nplot.plot_anat(t1_img_3d)

plt.show()
