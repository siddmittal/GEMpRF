import unittest 
import os
import xml.etree.ElementTree as ET
import shutil

class GemIndividualRunBidsHandling(unittest.TestCase):
    def get_cofig_dict(self):
        config_dict = {
            # Input data
            ".//fixed_paths/measured_data_filepath/filepath" : "this_is_BIDS_test_so_paths_will_be_computed_using_the_provided_BIDS_info", 
            ".//BIDS/basepath" : os.path.join(self.current_script_dir, 'testdata', 'bids_test', 'BIDS'), 
            ".//BIDS/append_to_basepath" : "derivatives, prfprepare", 
            ".//BIDS/analysis" : "01", 
            ".//BIDS/sub" : "dummy", 
            ".//BIDS/hemi" : "L",             
            ".//BIDS/individual/ses" : "001, 002", 
            ".//BIDS/individual/task" : "dummybar, dummyring", 
            ".//BIDS/individual/run" : "01", 
            


            # Results
            ".//fixed_paths/results/basepath" : self.temp_dir,
            ".//fixed_paths/results/custom_filename_postfix" : "",
            ".//fixed_paths/results/prepend_date" : "False",

            # Analysis Model
            ".//pRF_model_details/model" : "not_required",

            # Stimulus
            ".//stimulus/filepath" : "not_required",
            ".//stimulus/visual_field" : "not_required",
            ".//stimulus/width" : "not_required",
            ".//stimulus/height" : "not_required",

            # Search Space
            ".//search_space/visual_field" : "not_required",
            ".//search_space/nRows" : "not_required",
            ".//search_space/nCols" : "not_required",
            ".//search_space/min_sigma" : "not_required",
            ".//search_space/max_sigma" : "not_required",
            ".//search_space/nSigma" : "not_required",

            # Concatenated runs         
            ".//BIDS/concatenated/enable" : "False"
        }

        return config_dict

    def set_test_config_data(self, test_config_dict):
        from gem.configs.gem_xml_utils import XMLUtils

        # NOTE: Set configuration for test        
        #...enable BIDS attribute
        XMLUtils.update_xml_node_attribute(xml_filepath=self.temp_config_xml_file_path, 
                                       xpath=".//BIDS", 
                                       attribute_name="enable", 
                                       new_attribute_value='True')
        # ...change other values
        for key, value in test_config_dict.items():
            XMLUtils.update_xml_node_value(xml_filepath=self.temp_config_xml_file_path, 
                                       xpath=key,
                                       new_value=value)
        
        # Save the path to the modified config file for use in tests
        self.temp_config_xml_file_path = self.temp_config_xml_file_path


    # calls automatically before each test
    def setUp(self):        
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        gem_dir = os.path.dirname(self.current_script_dir)
        gem_parent_dir = os.path.dirname(gem_dir)
        os.sys.path.append(gem_parent_dir)
        os.sys.path.append(gem_dir)
        

        # Create a temporary results directory
        self.temp_dir = os.path.join(self.current_script_dir, "temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Make a copy of the default config file to the temp directory
        org_config_xml_file_path = os.path.join(gem_dir, 'configs', 'default_config', "default_config.xml")
        self.temp_config_xml_file_path = os.path.join(self.temp_dir, "test_bids_handler_config.xml")
        shutil.copyfile(org_config_xml_file_path, self.temp_config_xml_file_path)

    def test_gem_bids_handler_finds_correct_input_data_files(self):
        """
        Test case to verify if the GEM's BIDS handler can retreive the correct files.

        This test case performs the following steps:
        1. Runs the GemBidsHandler
        2. Retrieves the files information based on the provided BIDS configuration.

        """


        # arrange  
        from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
        from gem.data.bids_handler import GemBidsHandler     
        test_config_dict = self.get_cofig_dict()
        self.set_test_config_data(test_config_dict)   
        cfg = GEMpRFAnalysis.load_config(config_filepath=self.temp_config_xml_file_path) # load default config
        expected_measured_data_list = [
            os.path.join(self.current_script_dir, 'testdata', 'bids_test', 'BIDS', 'derivatives', 'prfprepare', 'analysis-01', 'sub-dummy', 'ses-001', 'func', 'sub-dummy_ses-001_task-dummybar_run-01_hemi-L_bold.nii.gz'),
            os.path.join(self.current_script_dir, 'testdata', 'bids_test', 'BIDS', 'derivatives', 'prfprepare', 'analysis-01', 'sub-dummy', 'ses-001', 'func', 'sub-dummy_ses-001_task-dummyring_run-01_hemi-L_bold.nii.gz')
        ]
                    
        # act
        measured_data_info_list = GemBidsHandler.get_input_filepaths(cfg.bids)
        measured_data_filepaths_list = [data[0] for data in measured_data_info_list]    

        # assert      
        # ...first we need to confirm that we have been able to retreive the correct number of files  
        assert len(expected_measured_data_list) == len(measured_data_filepaths_list), "The number of files retreived by GEM's BIDS Handler class do not match the expected number of files."

        # ...then we need to confirm that the filepaths are correct
        all_elements_true = all([
            expected == measured 
            for expected, measured in zip(expected_measured_data_list, measured_data_filepaths_list)
        ])
        assert all_elements_true, (
            f"Test Failed because the filepaths retreived by GEM's BIDS Handler class do not match the expected filepaths."
             f"\n\nGemBidsHandler retrieved list is {measured_data_filepaths_list}"             
             f"\n\nwhich is not equal to the Expected Measured Data List {expected_measured_data_list}"
        )


# To Run test to see the output of the print statements
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(GemIndividualRunBidsHandling)
    runner = unittest.TextTestRunner(buffer=False)
    runner.run(suite)

