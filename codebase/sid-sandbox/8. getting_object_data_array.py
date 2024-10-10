import numpy as np
import matplotlib.pyplot as plt
from typing import List

nRows_grid = 4
nCols_grid = 6

class ReceptiveFieldResponse:
    def __init__(self, x, y, timecourse):
        self.x = x
        self.y= y
        self.timecourse = timecourse


# Ordinary Least Squares (OLS)
class OLS:
    def __init__(self
                 , all_timcourses ):
        self.timecourses = all_timcourses
        

#### ----Program--------#####
dummy_receptive_field_data = []

for i in range(nRows_grid):
    for j in range(nCols_grid):
        y = i
        x = j
        dummy_timecourse = [1, 2]
        dummy_value = ReceptiveFieldResponse(x=x, y=y, timecourse=dummy_timecourse)
        dummy_receptive_field_data.append(dummy_value)

ols = OLS(dummy_receptive_field_data)     


## How do I extarct all timecourses present in the dummy_receptive_field_data
# Extract all timecourses using the map function
all_timecourses = list(map(lambda item: item.timecourse, dummy_receptive_field_data))

print("done")

         