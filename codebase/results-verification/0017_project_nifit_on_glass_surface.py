import json
import numpy as np
import nibabel as nib
import sys
# sys.path.append('/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/dlinhardt/pythonclass')
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")


from PRFclass import PRF
from os import path



y_signals_path = r"D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data\sub-sidtest_ses-001_task-bar_run-01_hemi-L_bold.nii.gz"
bold_response_img = nib.load(y_signals_path)
Y_signals_cpu = bold_response_img.get_fdata()

sub = 'sidtest'
ses = '001'
task = 'bar'
basePath="Y:\\data"
ana = PRF.from_docker('stimsim23', sub, ses, task, '01', analysis='03',baseP=basePath, method='vista')

print()
