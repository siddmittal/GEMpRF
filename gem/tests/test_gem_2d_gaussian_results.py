import unittest 
import os
import xml.etree.ElementTree as ET
import shutil

class GemAnalysisTests(unittest.TestCase):
    def get_cofig_dict(self):
        config_dict = {
            # Input data
            ".//BIDS/basepath" : "this_is_FIXED_FILEPATH_test_so_FIXED_filepats_specified_below", 
            ".//fixed_paths/measured_data_filepath/filepath" : os.path.join(self.current_script_dir, 'testdata/simulated/3n2', "sub-001_ses-3n2_task-prf_acq-normal_run-01_bold.nii.gz"), 

            # Results
            ".//fixed_paths/results/basepath" : self.temp_dir,
            ".//fixed_paths/results/custom_filename_postfix" : "",
            ".//fixed_paths/results/prepend_date" : "False",

            # Analysis Model
            ".//pRF_model_details/model" : "2d_gaussian",

            # Stimulus
            ".//stimulus/filepath" : os.path.join(self.current_script_dir, 'testdata/simulated/3n2', "sub-001_ses-3n2_task-prf_apertures.nii.gz"),
            ".//stimulus/visual_field" : "10",
            ".//stimulus/width" : "101",
            ".//stimulus/height" : "101",

            # Search Space
            ".//search_space/visual_field" : "13.5",
            ".//search_space/nRows" : "51",
            ".//search_space/nCols" : "51",
            ".//search_space/min_sigma" : "0.5",
            ".//search_space/max_sigma" : "5",
            ".//search_space/nSigma" : "8",

            # Concatenated runs
            ".//concatenated_runs/enable" : "False"            
        }

        return config_dict

    def set_test_config_data(self, test_config_dict):
        from gem.configs.gem_xml_utils import XMLUtils

        # NOTE: Set configuration for test        
        #...disable BIDS attribute
        XMLUtils.update_xml_node_attribute(xml_filepath=self.temp_config_xml_file_path, 
                                       xpath=".//BIDS", 
                                       attribute_name="enable", 
                                       new_attribute_value='False')
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
        self.temp_config_xml_file_path = os.path.join(self.temp_dir, "test_config.xml")
        shutil.copyfile(org_config_xml_file_path, self.temp_config_xml_file_path)

    def test_gem_current_results_are_same_as_before_on_simulated_data_location_3n2(self):
        """
        Test case to verify if the GEM estimation produces correct values.

        This test case performs the following steps:
        1. Runs the GEMpRFAnalysis using the provided configuration XML file.
        2. Retrieves the estimation data from the new results file.
        3. Retrieves the estimation data from the benchmark results file.
        4. Compares the new results with the benchmark results and calculates the maximum difference.
        5. Asserts that the maximum difference is less than or equal to 0.01.

        If the maximum difference exceeds the threshold, the test will fail.

        """

        # arrange  
        from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
        from gem.utils.gem_load_estimations import EstimationsUtils
        test_config_dict = self.get_cofig_dict()
        self.set_test_config_data(test_config_dict)
        
        # act
        GEMpRFAnalysis.run(self.temp_config_xml_file_path)
        new_results_data = EstimationsUtils.get_estimation_data(filepath=os.path.join(self.temp_dir, "sub-001_ses-3n2_task-prf_acq-normal_run-01_estimates.json"))
        benchmark_results_data = EstimationsUtils.get_estimation_data(filepath=os.path.join(self.current_script_dir, 'testdata/simulated/3n2', "2024-06-18_sub-001_ses-3n2_task-prf_acq-normal_run-01_estimates_[gem-benchmark_51x51x8].json"))
        max_difference = EstimationsUtils.compare_estimation_results(new_results=new_results_data, benchmark_results=benchmark_results_data)

        # assert
        self.assertLessEqual(max_difference, 0.01, f"Test Failed because the Maximum difference in New-Results vs. Benchmark is {max_difference}")


    def test_gem_predicts_average_3n2_for_simulated_location_3n2(self):
        """
        Test case to verify if GEM predicts the correct values for a simulated pRF (x=3, y=2, sigma=1)

        Steps:
        1. Set up the necessary test configuration data.
        2. Run GEMpRFAnalysis using the temporary configuration XML file.
        3. Retrieve the estimation data from the generated JSON file.
        4. Calculate the average values for the estimated 2D Gaussian parameters.
        5. Assert that the mean_Centerx0 value is between 2.9 and 3.1.
        6. Assert that the mean_Centery0 value is between 1.9 and 2.1.
        7. Assert that the mean_sigmaMajor value is between 0.9 and 1.1.
        """

        # arrange  
        from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
        from gem.utils.gem_load_estimations import EstimationsUtils
        test_config_dict = self.get_cofig_dict()
        self.set_test_config_data(test_config_dict)

        # act
        GEMpRFAnalysis.run(self.temp_config_xml_file_path)
        new_results_data = EstimationsUtils.get_estimation_data(filepath=os.path.join(self.temp_dir, "sub-001_ses-3n2_task-prf_acq-normal_run-01_estimates.json"))
        mean_Centerx0, mean_Centery0, mean_sigmaMajor = EstimationsUtils.get_avg_2d_gaussian_estimated_values(json_data=new_results_data)

        # assert
        assert mean_Centerx0>2.9 and mean_Centerx0 < 3.1, f"Test Failed because the estimated mean_Centerx0 = {mean_Centerx0} does not lie in the range 2.9 - 3.1"
        assert mean_Centery0>1.9 and mean_Centery0 < 2.1, f"Test Failed because the estimated mean_Centery0 = {mean_Centery0} does not lie in the range 1.9 - 2.1"
        assert mean_sigmaMajor>0.9 and mean_sigmaMajor < 1.1, f"Test Failed because the estimated mean_Centery0 = {mean_sigmaMajor} does not lie in the range 0.8 - 1.1"


# To Run test to see the output of the print statements
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(GemAnalysisTests)
    runner = unittest.TextTestRunner(buffer=False)
    runner.run(suite)

