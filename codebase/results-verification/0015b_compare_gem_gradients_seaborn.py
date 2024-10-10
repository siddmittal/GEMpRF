import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

df3 = pd.read_excel("D:/results/gradients-test/gradients-new-Estimations-[without-minus-sigma]_fx-51x51x8.xlsx", index_col=None)

coarse_est = df3['CoarseEsitmations']
refine_est = df3['RefinedEstimations']

coarse_result = []
refine_result = []


for idx in range(len(coarse_est)):
    coarse_values = ast.literal_eval(coarse_est[idx])
    refine_values = [float(val) for val in refine_est[idx][1:-1].split()] #ast.literal_eval(refine_est[idx])
    coarse_result.append(np.array(coarse_values))
    refine_result.append(np.array(refine_values))

coarse_muX = (np.array(coarse_result))[:, 0]
coarse_muY = (np.array(coarse_result))[:, 1]
coarse_sigma = (np.array(coarse_result))[:, 2]


# 3d points
coarse_points = []
for idx in range(len(coarse_result)):
    coarse_points.append(np.linalg.norm(coarse_result[idx]))

refine_points = []
for idx in range(len(refine_result)):
    refine_points.append(np.linalg.norm(refine_result[idx]))

# difference
difference = np.array(coarse_points) - np.array(refine_points)
indices_to_mark = np.where(difference > 10)[0]


# comparsion plot
plt.figure()
plt.plot(coarse_points)
plt.plot(refine_points)
plt.scatter(indices_to_mark, np.array(coarse_points)[indices_to_mark], color='red', label='Difference > 10')
plt.show()


print()    