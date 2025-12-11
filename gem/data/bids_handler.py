# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024 - 2025, Medical University of Vienna",
"@Desc    :   None",
        
"""

import json
import os
import re
import sys

from gem.data.diagnostic_bids_tree import DiagnosticBidsTree
from gem.data.gem_bids_concatenation_data_info import BidsConcatenationDataInfo
from gem.data.gem_stimulus_file_info import StimulusFileInfo
from gem.utils.logger import Logger

class GemBidsHandler:
    @classmethod
    def __get_stimulus_file_info(cls, stimulus_dir : str, stimulus_filename : str):
        match = re.search(r'task-([a-zA-Z0-9\-]+)', stimulus_filename) # regex to extract task name even if it contains hyphens
        if match:
            stimulus_task = match.group(1)
        else:
            raise ValueError(f"Stimulus Task name not found in the stimulus file name: {stimulus_filename}")
        
        stimulus_info = StimulusFileInfo(stimulus_dir, stimulus_filename, stimulus_task)
        return stimulus_info
    
    @classmethod
    def extract_value_from_bids_string(cls, bids_format_string : str, key :str):
        parts = bids_format_string.split('_')
        for part in parts:
            if '-' in part:
                found_key, found_value = part.split('-', 1) # split only on the first hyphen
                if found_key == key:
                    return found_value

    @classmethod
    def __print_bids_tree_summary(cls, base_path, file_extension, analysis_list, sub_list, ses_list, run_list, task_list, hemi_list, space_list, is_individual_run):
        # compact list formatter
        def _short(x, limit=4):
            if not x:
                return "-"
            if len(x) <= limit:
                return ",".join(map(str, x))
            return ",".join(map(str, x[:limit])) + ",..."

        tree_items = [
            f"base= {base_path}",
            f"ext= {file_extension}",
            f"analysis= {_short(analysis_list)}",
            f"sub= {_short(sub_list)}",
            f"ses= {_short(ses_list)}",
            f"run= {_short(run_list)}",
            f"task= {_short(task_list)}",
            f"hemi= {_short(hemi_list)}",
            f"space= {_short(space_list)}",
            f"run type= {'individual' if is_individual_run else 'concatenated'}",
        ]

        # print root in green
        Logger.print_green_message("\nSearching BIDS tree with following user filters:", print_file_name=False)

        # print as a nice tree
        for i, item in enumerate(tree_items):
            prefix = "└── " if i == len(tree_items) - 1 else "│── "
            print(prefix + item)

    @classmethod    
    def __get_matching_files(cls, base_path, append_to_basepath_list, analysis_list, sub_list, hemi_list, space_list, json_item, stimuli_dir_path, file_extension, is_individual_run):
        if is_individual_run: # in case of individual runs, the user may specify the multiple session/run values (but only one task) and therefore it could be a list
            ses_list = [element.strip() for element in json_item.get("ses").split(',')]
            run_list = [element.strip() for element in json_item.get("run").split(',')]        
            # # task_list = [element.strip() for element in json_item.get("task").split(',')]        
        else: # for concaenated run, only one session/run/task is allowed for each "concatenate_item" item
            ses_list = [json_item.get("ses").strip()]
            run_list = [json_item.get("run").strip()]        
            
        task_list = [json_item['task'].strip()]


        if append_to_basepath_list:
            base_path = os.path.join(base_path, *append_to_basepath_list)
        base_path = base_path.replace("\\", "/")

        # quick summary of the BIDS tree being searched
        cls.__print_bids_tree_summary(base_path, file_extension, analysis_list, sub_list, ses_list, run_list, task_list, hemi_list, space_list, is_individual_run)

        matching_files_info = []

        # analysis
        for analysis_dir_root, analysis_dirs, _ in os.walk(base_path):            
            analysis_dirs[:] = [d for d in analysis_dirs if d.startswith(('analysis')) and (d.split('-')[1] in analysis_list or 'all' in analysis_list)]
            for analysis_dir in analysis_dirs:
                analysis_id = analysis_dir.split('-')[1]
                analysis_path = os.path.join(analysis_dir_root, analysis_dir)

                # subjects
                sub_dir_root = analysis_path
                if not os.path.isdir(sub_dir_root):
                    continue                
                sub_dirs = os.listdir(sub_dir_root)
                sub_dirs[:] = [d for d in sub_dirs if d.startswith(('sub')) and os.path.isdir(os.path.join(sub_dir_root, d)) and (d.split('-')[1] in sub_list or 'all' in sub_list)]
                for sub_dir in sub_dirs:
                    sub_path = os.path.join(sub_dir_root, sub_dir)
                    
                    # sessions for this subject
                    ses_dir_root = sub_path
                    if not os.path.isdir(ses_dir_root):
                        continue                    
                    ses_dirs = os.listdir(ses_dir_root)
                    ses_dirs[:] = [d for d in ses_dirs if d.startswith(('ses')) and os.path.isdir(os.path.join(ses_dir_root, d)) and (d.split('-')[1] in ses_list or 'all' in ses_list)]
                    for ses_dir in ses_dirs:
                        ses_path = os.path.join(ses_dir_root, ses_dir)

                        # func
                        func_dir_root = ses_path
                        if not os.path.isdir(func_dir_root):
                            continue                        
                        func_dirs = os.listdir(func_dir_root)
                        func_dirs[:] = [d for d in func_dirs if d.startswith(('func'))]
                        for func_dir in func_dirs:
                            func_path = os.path.join(func_dir_root, func_dir)
                            for _, _, files in os.walk(func_path):

                                # files level
                                for file in files:
                                    # .nii.gz or .gii
                                    if (file.endswith(file_extension) or file_extension.lower() == 'both') \
                                        and (file.endswith('_bold.nii.gz') or file.endswith('_bold.func.gii')):
                                        # parts = file.split('_')
                                        filename_without_ext = (file.split('.'))[0]
                                        parts = filename_without_ext.split('_')
                                        filename_info_dict = {}
                                        for part in parts:
                                            if '-' in part:
                                                key, value = part.split('-', 1) # split only on the first hyphen
                                                filename_info_dict[key] = value
                                        
                                        filename_info_dict['analysis'] = analysis_id # also add analysis id to the filename_parts

                                        sub_part = filename_info_dict.get('sub')
                                        ses_part = filename_info_dict.get('ses')
                                        task_part = filename_info_dict.get('task')
                                        run_part = filename_info_dict.get('run')
                                        hemi_part = filename_info_dict.get('hemi')
                                        space_part = filename_info_dict.get('space', None) # if filename does not contain space part, consider it None

                                        if ('all' in sub_list or sub_part in sub_list):
                                            if ('all' in ses_list or ses_part in ses_list):
                                                # # if ('all' in task_list or task_part in task_list):
                                                if (task_part in task_list):
                                                    if ('all' in run_list or run_part in run_list):
                                                        if ('all' in hemi_list or hemi_part in hemi_list):
                                                            if (space_part is None or 'all' in space_list or space_part in space_list):
                                                                stimulus_info = cls.get_stimulus_info(stimuli_dir_path, task_part)
                                                                # NOTE: Replaced "func_dir_root" with "func_path" to get the correct path
                                                                matching_files_info.append(
                                                                    tuple((os.path.join(func_path, file), filename_info_dict, stimulus_info)))
        # BIDS Tree Diagnostics:
        try:
            diag = DiagnosticBidsTree.collect_bids_diagnostics(base_path)
            DiagnosticBidsTree.print_bids_diagnostic_tree(diag, analysis_list, sub_list, ses_list, run_list, task_list, hemi_list, space_list, file_extension)
        except Exception as e:
            Logger.print_red_message(f"Could not collect or print BIDS diagnostic tree: {e}", print_file_name=False)

        return matching_files_info

    @classmethod
    def get_stimulus_info(cls, stimulus_dir, stimulus_name):
        available_stimuli_files = [i for i in os.listdir(stimulus_dir) if i.endswith('.nii.gz')]
        stimulus_for_this_task =  [stimulus_file for stimulus_file in available_stimuli_files if stimulus_name == cls.extract_value_from_bids_string(bids_format_string = stimulus_file, key = 'task')]                                                                        
        stimulus_info = cls.__get_stimulus_file_info(stimulus_dir, stimulus_for_this_task[0])   
        return stimulus_info     

    @classmethod
    def get_non_bids_stimulus_info(cls, cfg):
        stimulus_filepath = cfg.fixed_paths['stimulus_filepath']
        if not os.path.isfile(stimulus_filepath):
            Logger.print_red_message(f"Stimulus file not found at the specified path: {stimulus_filepath}", print_file_name=False)
            sys.exit(1)
        stimulus_dir = os.path.dirname(stimulus_filepath)
        stimulus_name = os.path.basename(stimulus_filepath)

        stimulus_info = cls.__get_stimulus_file_info(stimulus_dir, stimulus_name)
        return stimulus_info   

    @classmethod
    def get_input_filepaths(cls, bids_config : json, stimuli_dir_path : str):
        base_path = bids_config.get("basepath")
        file_extension = bids_config.get("input_file_extension")['#text']
        if file_extension is None or file_extension.lower() not in ['.nii.gz', '.gii', 'both']:
            Logger.print_red_message(f"Invalid or missing input_file_extension in BIDS config inside configuration XML file. Supported extensions are .nii.gz/.gii/both", print_file_name=False)
            sys.exit(1)

        # common information for both "individual" and "concantenated" analysis
        append_to_basepath_list = [element.strip() for element in bids_config.get("append_to_basepath").split(',')]
        analysis_list = [element.strip() for element in bids_config.get("analysis").split(',')]
        sub_list = [element.strip() for element in bids_config.get("sub").split(',')]
        hemi_list = [element.strip() for element in bids_config.get("hemi").split(',')]
        space_list = [element.strip() for element in bids_config.get("space")['#text'].split(',')]

        # this could be a single list (in case of individual analysis) or a list-of-list (in case of concatenated analysis)
        input_src_files = []

        # check if "Individual" or "Concatenated" tasks are specified
        if bids_config['@enable'] == "True" and bids_config['@run_type'].lower() == "concatenated":
            concatenate_items_list = bids_config.get("concatenated").get("concatenate_item")
            for i, concatenate_item in enumerate(concatenate_items_list, start=1):
                Logger.print_blue_message(f"\n[ Gathering information for Concatenated block - {i} ]", print_file_name=False)
                src_files = GemBidsHandler.__get_matching_files(base_path, append_to_basepath_list, analysis_list, sub_list, hemi_list, space_list, concatenate_item, stimuli_dir_path, file_extension, is_individual_run=False)
                input_src_files.append(src_files)

        else:
            individual_item = bids_config.get("individual")
            input_src_files = GemBidsHandler.__get_matching_files(base_path, append_to_basepath_list, analysis_list, sub_list, hemi_list, space_list, individual_item, stimuli_dir_path, file_extension, is_individual_run=True)
            input_src_files = sorted(input_src_files, key=lambda x: (x[0], x[1]['sub'], x[1]['ses'], x[1]['run']))

        return input_src_files
    
    @classmethod
    def inputpath2resultpath(self, bids_config : json, input_file_info : tuple, analysis_id : str = None):
        filename = os.path.basename(input_file_info[0])
        if filename.endswith('.nii.gz'):
            filename_without_extension = filename.split('_bold.nii.gz')[0]
        elif filename.endswith('.gii'):
            filename_without_extension = filename.split('_bold.func.gii')[0]
        
        # get BIDS info
        base_path = bids_config.get("basepath")        
        if (analysis_id is None or analysis_id == ""):
            analysis = input_file_info[1]['analysis'] #  use the analysis id from the input file info
        else:
            analysis = analysis_id
        
        sub = input_file_info[1]['sub']
        ses = input_file_info[1]['ses']
        # # task = input_file_info[1]['task']
        # # run = input_file_info[1]['run']
        # # hemi = input_file_info[1]['hemi']
            
        base_path = os.path.join(base_path, 'derivatives', 'prfanalyze-gem', f'analysis-{analysis}', f'sub-{sub}', f'ses-{ses}')    
        ## os.makedirs(base_path, exist_ok=True)
        filepath = os.path.join(base_path, filename_without_extension + '_estimates.json')

        return filepath

    @classmethod    
    def update_filename(cls, filename, info_dict):    
        parts = filename.split('_')  # Split filename by underscores to handle different sections
        
        for i in range(len(parts)):
            part = parts[i]
            for key, value in info_dict.items():
                if part.startswith(f"{key}-"):
                    parts[i] = f"{key}-{value}"
                    break  # Stop after the first match to avoid overwriting subsequent parts

        updated_filename = '_'.join(parts)  # Recombine parts into the updated filename
        return updated_filename    

    @classmethod
    def get_concatenated_result_filepath(cls, bids_config : json, filepath : str, concatenation_result_info : dict):
        if filepath.endswith('.nii.gz'):
            filename_without_extension = os.path.basename(filepath).split('_bold.nii.gz')[0]
        elif filepath.endswith('.gii'):
            filename_without_extension = os.path.basename(filepath).split('_bold.func.gii')[0]

        concatenated_result_filename = GemBidsHandler.update_filename(filename_without_extension, concatenation_result_info)

        base_path = bids_config.get("basepath")
        results_anaylsis_id = bids_config["results_anaylsis_id"]["#text"]

        sub = concatenation_result_info['sub']
        ses = concatenation_result_info['ses']        
        
        base_path = os.path.join(base_path, 'derivatives', 'prfanalyze-gem', f'analysis-{results_anaylsis_id}', f'sub-{sub}', f'ses-{ses}')    
        # os.makedirs(base_path, exist_ok=True)

        filepath = os.path.join(base_path, concatenated_result_filename + '_estimates.json')

        return filepath