import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

y_idx = 11

coarse_error = nib.load(r"D:\results\gradients-test\saved-debug-data\11_coarse_error.nii.gz").get_fdata()
coarse_error_gradients = nib.load(r"D:\results\gradients-test\saved-debug-data\11_error_gradients.nii.gz").get_fdata()
fex_error = nib.load(r"D:\results\gradients-test\saved-debug-data\11_error_fex.nii.gz").get_fdata()
fex_error_gradients = nib.load(r"D:\results\gradients-test\saved-debug-data\11_error_gradients_fex.nii.gz").get_fdata()




print()
