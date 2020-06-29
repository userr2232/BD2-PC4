import pandas as pd
import numpy as np

encodings = pd.read_csv("encodings/complete.txt", header=None)

# iloc para traer filas y columnas, like an matrix for data
data = encodings.iloc[ : ,1:129]
data.head(5933)

# definir querys
N = [5000,180,3258]
Qx = []
for i in N:
  Qx.append(data.iloc[i])

dataT = data.drop(N, axis=0)

#definir la medida de distancia euclediana
DE_l2 = lambda x,y : sum((x-y)**2)**0.5
DM_l1 = lambda x,y : sum(abs(x-y))

"""Algoritmo de busqueda KNN"""

def knnSearch(data, Q, k):
  result = []
  for index,row in data.iterrows():
      d = DE_l2(Q, row)
      result.append((index,d))
  result.sort(key = lambda tup: tup[1])
  return result[:k]

#evaluar algortmo
for n_query in range(len(N)):
  for k in [2,4,8,16,32]:
    p1 = knnSearch(dataT, Qx[n_query],k)
    target = encodings.iloc[N[n_query],129]
    count = 0
    for i,d in p1:
      if target == encodings.iloc[i,129]:
        count +=1
    print("Elementos recuperados", len(p1))
    print("Presicion: ", count/len(p1))
    print(encodings.iloc[[x for x,y in p1], 129])
    print(target)

data.head(5933)

import heapq

def knnSearchHeap(data, Q, k):
    result = []
    for index, row in data.iterrows():
      d = DE_l2(Q, row)   
      heapq.heappush(result, (-d, index))
    while len(result) > k:
      heapq.heappop(result)    
    result = [ heapq.heappop(result) for _ in range(len(result))]
    result = list(reversed(result))
    return result

# evaluar algoritmo
for n_query in range(len(N)):
  for k in [2,4,8,16,32]:
      result = knnSearchHeap(dataT, Qx[n_query], k)
      target = encodings.iloc[N[n_query], 129]
      cont = 0
      for d,i  in result:
          if target == encodings.iloc[i, 129]:
              cont += 1
      print("Precision:", cont/len(result)) 
      print(encodings.iloc[[x for y,x in result], 129])

print(encodings)

from rtree import index
from sklearn.decomposition import PCA

p = index.Property()
p.dimension = 32
p.buffering_capacity = 5
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index(properties=p)

pca = PCA(n_components=32);
print(data.shape)


pca_data = pca.fit_transform(data);
k = 16
q = pca_data[5];
print("target", encodings.iloc[5,129]);
np.delete(pca_data, 0, 0);

for i in range(len(pca_data)):
  x = pca_data[i];
  idx.insert(i, (*x, *x));

res = list(idx.nearest(coordinates=(*q, *q), num_results=k));
print("results", encodings.iloc[res,129]);