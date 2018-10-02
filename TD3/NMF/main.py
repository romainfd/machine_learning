import numpy as np
import pandas as pd
from nmf import *
import matplotlib.pyplot as plt
with open('NMF/data/nyt_data.txt') as f:
    documents = f.readlines()
documents = [x.strip().strip('\n').strip("'") for x in documents]

# contains vocabs with rows as index
with open('NMF/data/nyt_vocab.dat') as f:
    vocabs = f.readlines()
vocabs = [x.strip().strip('\n').strip("'") for x in vocabs]

'''create matrix X'''
numDoc = 1000
numWord = 3012
X = np.zeros([numWord,numDoc],dtype=float)

for col in range(len(documents[:1000])):
    for row in documents[col].split(','):
        X[int(row.split(':')[0])-1,col] = float(int(row.split(':')[1]))

X=X+np.ones((numWord,numDoc))*0.00000001

rank = 25
W,H,d_iter=nmf_factor(X,rank)

fig= plt.figure(figsize = (15,6))
ax = fig.add_subplot(1,1,1)
ax.plot(range(100),d_iter[:100])
plt.title('Plot of euclidean norm objective in 100 iterations')
plt.ylabel('$||X-WH||$')
plt.xlabel('iteration $t$')
plt.show()


# ### b. Ten words with the largest weight.

'''normalize each column to sum to zero'''
W_normed = W / np.sum(W,axis=0)


'''for each column of W, list the 10 words having the largest weight and show the weight'''
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)
vList = []

for topic in range(rank):
    v = pd.DataFrame(vocabs)
    v[1] = W_normed[:,topic].round(6)
    v = v.sort_values([1, 0], ascending=[0,1]).rename(index=int, columns={0: "Topic {}".format(topic+1), 1: "Weight"}).head(10)
    v = v.reset_index(drop=True)
    vList.append(v)

for num in [5,10,15,20,25]:
    print('\n',(pd.concat(vList[num-5:num], axis=1)),'\n')
