# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:15:29 2024

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Medical University of Vienna",
"@Desc    :   None",
        
"""
import numpy as np
import cupy as cp

from gem.model.prf_model import PRFModel
from gem.model.selected_prf_model import SelectedPRFModel

# NOTE: Implement the methods/properties, that are made mandatory by the PRFModel class
class PRFDoGModel(PRFModel):
    """Extract text from an email."""
    def create_model(self, model_name: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        pass

    def extract_text(self, full_file_path: str) -> dict:
        """Overrides FormalParserInterface.extract_text()"""
        pass

    @property
    def model_type(self) -> SelectedPRFModel:
        """Overrides FormalParserInterface.model_type()"""
        return self.__model_type     