import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df1 = pd.read_excel("D:/results/gradients-test/gradients-new-with_fx-51x51x8.xlsx", index_col=None)
df1 = pd.read_excel("D:/results/gradients-test/gradients-new-with-[wihtout-minus-sigma]_fx-51x51x8.xlsx", index_col=None)
df2 = pd.read_excel("D:/results/gradients-test/gradients-new-without_fx-51x51x8.xlsx", index_col=None)

# boxplot for all in one
plt.figure()
sns.boxplot(data=df1, x='param', y='value', hue='type')

# lineplots for all in one ---- but this is not helpful, everything is overlayed
plt.figure()
sns.lineplot(data=df1, x=df1.index, y='value', hue='type')

# Seperate Line plots for e, de_dx, and de_dy
# ...separate data for e, de_dx, and de_dy
e_data = df1[df1['param'] == 'e']
de_dx_data = df1[df1['param'] == 'de_dx']
de_dy_data = df1[df1['param'] == 'de_dy'] 

#...index
e_data['index'] = range(len(e_data))
de_dx_data['index'] = range(len(de_dx_data))
de_dy_data['index'] = range(len(de_dy_data))

plt.figure()
sns.lineplot(x=e_data.index, y="value", data=e_data, hue="type")

plt.figure()
sns.lineplot(x=de_dx_data.index, y='value', data=de_dx_data, hue='type')


plt.figure()
sns.lineplot(x=de_dy_data.index, y='value', data=de_dy_data, hue='type')

# sns.lineplot(data=df1, x="param", y="value", hue="type")
plt.show()

print()