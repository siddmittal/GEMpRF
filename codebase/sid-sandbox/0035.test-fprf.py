import nibabel as nib
import numpy as np
# from cni_tlbx import HGR
import matplotlib.pyplot as plt
from Local_HGR import HGR
import json

def write_to_file( filepath, data):
    # Convert the list of dictionaries to a JSON string
    json_string = json.dumps(data, indent=4)

    # Write the JSON data to the file
    with open(filepath, 'w') as json_file:
        json_file.write(json_string)


def args2jsonEntry( muX, muY, sigma, r2, signal):
    json_entry = {
                "Centerx0": muX,
                "Centery0": muY,
                "Theta": 0,
                "sigmaMajor": sigma,
                "sigmaMinor": 0,
                "R2": r2,
                "modelpred": signal
            }
    return json_entry

# Convert NumPy arrays to lists in the entire dictionary
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError

def convert_dict_to_list(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = convert_dict_to_list(value)
        elif isinstance(value, np.ndarray):
            d[key] = value.tolist()
    return d


##############################
output_analysis = '03'
subjects = ['001', '002']
sessions = ['001', '002', '003']

for sub in subjects:
    for ses in sessions:
        #stim_NIFTI = nib.load("D:/results/docker-analysis/01-test/BIDS/stimuli/sub-001_ses-20200320_task-prf_apertures.nii.gz")
        stim_NIFTI = nib.load("Y:/data/tests/oprf_test/BIDS/derivatives/prfprepare/analysis-01/sub-001/stimuli/sub-001_ses-001_task-prf_apertures.nii.gz")

        # data_NIFTI = nib.load("D:/results/docker-analysis/01-test/BIDS/sub-001/ses-20200320/func/sub-001_ses-20200320_task-prf_acq-normal_run-01_bold.nii.gz")
        data_NIFTI = nib.load(f"Y:/data/tests/oprf_test/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-prf_acq-normal_run-01_bold.nii.gz")
        output_file_path = f"Y:/data/tests/oprf_test/BIDS/derivatives/prfanalyze-fprf/analysis-{output_analysis}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-prf_run-01_hemi-BOTH_estimates.json"

        data = np.array(data_NIFTI.dataobj)
        data  = np.squeeze(data, axis=(1, 2))
        data = data.T


        stimulus = np.array(stim_NIFTI.dataobj)
        stimulus = np.squeeze(stimulus, axis=2)
        stimulus = (np.reshape(stimulus, (101 * 101, 300))).T

        shrink = 6
        parameters = {
            'fwhm': 0.15,
            'eta': 0.1,
            'f_sampling': 1 / 1,  # 1/TR
            'r_stimulus': 101,  # stimulus size is 150x150
            'n_features': 300,  # number of tilings
            'n_gaussians': 5,  # number of Gaussians within a tile
            'n_voxels': 5000 # data.shape[1]  # Assuming 'data' is a NumPy array
        }

        hgr = HGR(parameters)  
        hgr.ridge(data, stimulus)
        results = hgr.get_parameters()

        # JSON file
        muX = results['mu_x']
        muY = results['mu_y']
        sigma = results['sigma']

        # Write to File
        json_data_results_with_r2 = []
        for x, y, sig in zip(muX, muY, sigma):
            json_entry = args2jsonEntry(x, y, sig, 0, [0])    
            json_data_results_with_r2.append(json_entry)    
        write_to_file(output_file_path, data=json_data_results_with_r2) # TODO: Uncomment this to write the results


        # # JSON file
        # output_file_path = "Y:/data/tests/oprf_test/BIDS/derivatives/prfanalyze-fprf/analysis-01/sub-002/ses-001/sub-002_ses-001_task-prf_run-01_hemi-BOTH_estimates.json"

        # # Convert the dictionary recursively
        # results = convert_dict_to_list(results)

        # # Write the debug output to a JSON file
        # with open(output_file_path, 'w') as json_file:
        #     json.dump(results, json_file, indent=4)

print()