import json
import os
from enum import Enum
import xmltodict

class RunType(Enum):
    ANALYSIS = 1
    SIMULATION = 2

class ConfigurationWrapper:
    config_data = None  # Class-level attribute to store the configuration data


    # # # NOTE: to be deleted once XML works
    # # @classmethod
    # # def _read_json_file(cls, run_type: RunType):
    # #     script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
        
    # #     if run_type == RunType.ANALYSIS:
    # #         config_file_path = os.path.join(script_dir, "analysis_configs", "config.json")  # Construct the path to config.json
    # #     elif run_type == RunType.SIMULATION:
    # #         config_file_path = os.path.join(script_dir, "simulation_configs", "config.json")  # Construct the path to config.json
    # #     else:
    # #         raise ValueError("Correct configuration cannot be loaded as the configuration type is not correctly passed")

    # #     if os.path.isfile(config_file_path):
    # #         with open(config_file_path, "r") as config_file:
    # #             cls.config_data = json.load(config_file)
    # #     else:
    # #         print("Config file not found. Make sure it's located in the same directory as the script.")


    @classmethod
    def __read_xml_file(cls, run_type: RunType = None, config_filepath : str = None):
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
        
        if config_filepath is not None:
            config_file_path = config_filepath
        else:
             config_file_path = os.path.join(script_dir, "default_config", "default_config.xml") # default config file path

        # # if run_type == RunType.ANALYSIS:
        # #     config_file_path = os.path.join(script_dir, "analysis_configs", "config.xml")  # Construct the path to config.xml
        # # elif run_type == RunType.SIMULATION:
        # #     config_file_path = os.path.join(script_dir, "simulation_configs", "config.xml")  # Construct the path to config.xml
        # # else:
        # #     raise ValueError("Correct configuration cannot be loaded as the configuration type is not correctly passed")

        if os.path.isfile(config_file_path):
            with open(config_file_path) as xml_file:            
                config_data_dict = xmltodict.parse(xml_file.read())
                xml_file.close()            

                 # XML to JSON data conversion
                str_config_data = json.dumps(config_data_dict)
                json_config_data = json.loads(str_config_data)
                cls.config_data = json_config_data.get("root")
        else:
            print("Config file not found. Make sure it's located in the same directory as the script.")    

    @classmethod
    def parse_attrs(cls, node, types=None):
        """Extracts attributes (removes '@'), converts values using types dict if provided."""
        if node is None:
            return {}
        types = types or {}
        return {
            k.lstrip("@"): types.get(k.lstrip("@"), str)(v)
            for k, v in node.items()
            if k.startswith("@")
        }  

    @classmethod
    def load_configuration(cls, run_type: RunType = None, config_filepath : str = None):
        # if cls.config_data is None: #NOTE: This sometimes lead to unpredictable results, to be on safe side, always load the configuration
        cls.__read_xml_file(run_type, config_filepath) 
            # # cls._read_json_file(run_type)

        # path_to_append
        cls.path_to_append = cls.config_data.get("path_to_append")

        # Input data source
        # ...BIDS
        cls.bids = (cls.config_data.get("input_datasrc")).get("BIDS") # sample: {'@enable': 'True', 'basepath': 'D:/results/fmri/generic/', 'analysis': None, 'sub': None, 'ses': None}
        # ...Non-BIDS i.e. direct filepaths
        cls.fixed_paths = (cls.config_data.get("input_datasrc")).get("fixed_paths")

        # result_file_path
        cls.result_file_path = cls.config_data.get("result_file_path")        

        # results
        #cls.results = cls.config_data.get("results")
        cls.results = (cls.config_data.get("input_datasrc")).get("fixed_paths").get("results")

        # PRF Model
        cls.pRF_model_details = cls.config_data.get("pRF_model")

        # Stimulus
        cls.stimulus = cls.config_data.get("stimulus")
        
        # Measure Data
        cls.measured_data = cls.config_data.get("measured_data")
        
        # Search Space
        cls.search_space = cls.config_data.get("search_space")  

        # Concatenated runs
        cls.concatenated_runs = cls.config_data.get("concatenated_runs")        

        # Test Space
        cls.test_space = cls.config_data.get("test_space")

        # Noise
        cls.noise = cls.config_data.get("noise")           

        # GPU
        cls.gpu = cls.config_data.get("gpu")           

        # Enable/disable Refine fitting  
        refine_fitting_node_params = cls.parse_attrs(cls.config_data.get("refine_fitting", {}), {
            "enable": lambda v: v.lower() == "true",
            "refinefit_on_gpu": lambda v: v.lower() == "true"
        })
        cls.refine_fitting_enabled = refine_fitting_node_params['enable']
        cls.is_refinefit_on_gpu = refine_fitting_node_params['refinefit_on_gpu']
     
        # Optional analysis params
        opt_node = cls.search_space.get("optional_analysis_params", {})
        cls.optional_analysis_params = cls.parse_attrs(opt_node, {
            "enable": lambda v: v.lower() == "true",
            "filepath": str
        })

        # ...subnodes of optional_analysis_params
        for key in ["hrf", "sigmas", "spatial_grid_xy"]:
            subnode = opt_node.get(key, {})  # <--- get from original XML dict
            cls.optional_analysis_params[key] = cls.parse_attrs(subnode, {
                "use_from_file": lambda v: v.lower() == "true",
                "key": str
            })   

        # default spatial grid
        cls.default_spatial_grid = cls.parse_attrs(cls.search_space.get("default_spatial_grid"), {
            "visual_field_radius": float,
            "num_horizontal_prfs": int,
            "num_vertical_prfs": int
        })

        # default sigmas
        cls.default_sigmas = cls.parse_attrs(cls.search_space.get("default_sigmas"), {
            "num_sigmas": int,
            "min_sigma": float,
            "max_sigma": float
        })

        print("Successfully read the config file. ")
        if cls.optional_analysis_params['enable']:
            print(f"Analysis params from file: {', '.join(k for k,v in cls.optional_analysis_params.items() if isinstance(v, dict) and v.get('use_from_file', False)) or 'None'}")

# Usage:
if __name__ == "__main__":
    ConfigurationWrapper.load_configuration(RunType.ANALYSIS)

    # Access the configuration parameters using the class itself
    print(f"Visual Field Size: {ConfigurationWrapper.path_to_appended}")
    print(f"Stimulus: {ConfigurationWrapper.stimulus}")
