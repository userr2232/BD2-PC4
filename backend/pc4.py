import pandas as pd
import numpy as np
import time

encodings = pd.read_csv("encodings/complete.txt", header=None)

data = encodings.iloc[ : ,1:129]
data.head(5933)

# definir querys
N = [3000]
Qx = []
for i in N:
  Qx.append(data.iloc[i])

dataT = data.drop(N, axis=0)

# Definir la medida de distancia euclediana y manhattan
DE_l2 = lambda x,y : sum((x-y)**2)**0.5
DM_l1 = lambda x,y : sum(abs(x-y))

"""Algoritmo de busqueda KNN"""
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

def knnSearchHeap_MD(data, Q, k):
    result = []
    for index, row in data.iterrows():
      d = DM_l1(Q, row)   
      heapq.heappush(result, (-d, index))
    while len(result) > k:
      heapq.heappop(result)    
    result = [ heapq.heappop(result) for _ in range(len(result))]
    result = list(reversed(result))
    return result

def knnSearch(data, Q180180, k):
  result = []
  for index,row in data.iterrows():
      d = DE_l2(Q, row)
      result.append((index,d))
  result.sort(key = lambda tup: tup[1])
  return result[:k]

#evaluar knnSearch
""""
for n_query in range(len(N)):
  for k in [4,8,16]:
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
"""

# evaluar KnnSearchHeap
print("ED")
for n_query in range(len(N)):
  for k in [4,8,16]:
    print("k: ",k)
    result = knnSearchHeap(dataT, Qx[n_query], k)
    target = encodings.iloc[N[n_query], 129]
    cont = 0
    for d,i  in result:
      if target == encodings.iloc[i, 129]:
          cont += 1
    print("Precision:", cont/len(result)) 
    #print(encodings.iloc[[x for y,x in result], 129])

#evaluar knnHeap_MD
print("MD")
for n_query in range(len(N)):
  for k in [4,8,16]:
    print("k: ",k)
    result = knnSearchHeap_MD(dataT, Qx[n_query], k)
    target = encodings.iloc[N[n_query], 129]
    cont = 0
    for d,i  in result:
      if target == encodings.iloc[i, 129]:
          cont += 1
    print("Precision:", cont/len(result)) 

#evaluar tiempos    
size_data = [100,200,400,800,1600,3200,6400,12800]

for i in size_data:
  data_aux = dataT.iloc[:i,:]
  st = time.time()
  p1 = knnSearchHeap(data_aux,Qx[0],4)
  ft = time.time()
  print(i,": ",ft-st)


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
