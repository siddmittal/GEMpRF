# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Medical University of Vienna",
"@Desc    :   None",
        
"""

from gem.data.gem_stimulus_file_info import StimulusFileInfo

class BidsConcatenationDataInfo:
    def __init__(self, input_filepaths_to_be_concatenated : list, input_data_info_to_be_concatenated_data_info : list, all_stimulus_info : StimulusFileInfo, concatenation_result_data_info : dict, concatenation_result_filepath : str) -> None:
        # information about each input data involved in concatenation
        self.filepaths_to_be_concatenated = input_filepaths_to_be_concatenated
        self.input_data_info_to_be_concatenated = input_data_info_to_be_concatenated_data_info
        self.all_stimuli_info = all_stimulus_info 

        # information about the concatenation results
        self.concatenation_result_data_info = concatenation_result_data_info
        self.concatenation_result_filepath = concatenation_result_filepath
        

    @classmethod
    def compare_and_merge_data_info_dicts(cls, data_info_dictionaries : list):
        # Get all keys present in any dictionary
        all_keys = set().union(*data_info_dictionaries)

        result_dict = data_info_dictionaries[0].copy() #{}

        for key in all_keys:
            # Get all values for the current key from all dictionaries
            values = [d[key] for d in data_info_dictionaries if key in d]
            
            # Check if all values are the same
            if len(set(values)) > 1:
                # Concatenate all different values and add to result dictionary
                result_dict[key] = ''.join(values)

        return result_dict