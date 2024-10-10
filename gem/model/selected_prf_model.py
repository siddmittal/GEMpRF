# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Medical University of Vienna",
"@Desc    :   None",
        
"""

from enum import Enum            
class SelectedPRFModel(Enum):
    GAUSSIAN = 0
    DoG = 1
    CSS = 2
    NoneType = 3