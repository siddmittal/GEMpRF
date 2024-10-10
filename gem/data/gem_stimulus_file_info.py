# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Medical University of Vienna",
"@Desc    :   None",
        
"""

class StimulusFileInfo:
    def __init__(self, stimulus_dir : str, stimulus_filename : str, stimulus_task : str) -> None:
        self.stimulus_dir = stimulus_dir
        self.stimulus_filename = stimulus_filename
        self.stimulus_task = stimulus_task