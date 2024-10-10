#%%---dice is no longer fair
from pandas import DataFrame
d=DataFrame(index=[(i,j) for i in range(1,7) for j in range(1,7)], columns=['sm','d1','d2','pd1','pd2','p'])

d.d1=[i[0] for i in d.index]
d.d2=[i[1] for i in d.index]
d.sm=list(map(sum,d.index))
d.loc[d.d1<=3,'pd1']=1/9.
d.loc[d.d1 > 3,'pd1']=2/9.
d.pd2=1/6.
d.p = d.pd1 * d.pd2


d.groupby('sm')['p'].sum()


print("done")



#%%----sum of the dice equals seven
d={(i,j):i+j for i in range(1,7) for j in range(1,7)}

from collections import defaultdict
dinv = defaultdict(list)

for i,j in d.items(): # basicall "i" contains the pair information and "j" contains the sum
    dinv[j].append(i)

X={i:len(j)/36. for i,j in dinv.items()}
print(X)
print("done")    

#%%
# Next program