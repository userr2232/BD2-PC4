import os
import pandas as pd
import numpy as np
import heapq
from rtree import index
from sklearn.decomposition import PCA

class KNN():
  dimension = 32;

  def __init__(self):
    self.encodings = pd.read_csv("encodings/all.txt", header=None);
    # iloc para traer filas y columnas, like an matrix for data
    self.data = self.encodings.iloc[ : ,1:129];
    self.rtreeBuild();

  def rtreeBuild(self):
    p = index.Property();
    p.dimension = KNN.dimension;
    p.buffering_capacity = 5;
    p.dat_extension = 'data';
    p.idx_extension = 'index';
    self.idx = index.Index(properties=p);
    self.pca = PCA(n_components=KNN.dimension);
    self.pca_data = self.pca.fit_transform(self.data);
    for i in range(len(self.pca_data)):
      x = self.pca_data[i];
      self.idx.insert(i, (*x, *x));

  def knnSeqTest(self):
    # definir querys
    N = [5000,180,3258];
    Qx = [];
    for i in N:
      Qx.append(self.data.iloc[i]);
    dataT = self.data.drop(N, axis=0);

    #evaluar knnSearch
    for n_query in range(len(N)):
      for k in [2, 4, 8, 16, 32]:
        p1 = self.knnSearch(dataT, Qx[n_query], k)
        target = self.encodings.iloc[ N[n_query], 129 ]
        count = 0
        for i, d in p1:
          if target == self.encodings.iloc[i,129]:
            count +=1
        print("Elementos recuperados", len(p1))
        print("Presicion: ", count/len(p1))
        print(self.encodings.iloc[[x for x, y in p1], 129])
        print(target)

    # evaluar knnSearchHeap
    for n_query in range(len(N)):
      for k in [2, 4, 8, 16, 32]:
        result = knnSearchHeap(dataT, Qx[n_query], k)
        target = self.encodings.iloc[N[n_query], 129]
        cont = 0
        for d,i  in result:
          if target == self.encodings.iloc[i, 129]:
            cont += 1
        print("Precision:", cont/len(result)) 
        print(self.encodings.iloc[[x for y, x in result], 129])

  # definir la medida de distancia euclidiana
  @staticmethod
  def DE_l2(x, y):
    return sum((x - y)**2)**0.5;

  @staticmethod
  def DM_l1(x, y):
    return sum(abs(x - y));

  """Algoritmo de busqueda KNN"""

  @staticmethod
  def knnSearch(data, Q, k):
    result = []
    for index, row in data.iterrows():
        d = KNN.DE_l2(Q, row)
        result.append((index, d))
    result.sort(key = lambda tup: tup[1])
    return result[:k]

  @staticmethod
  def knnSearchHeap(data, Q, k):
    result = [];
    for index, row in data.iterrows():
      d = DE_l2(Q, row);
      heapq.heappush(result, (-d, index));
    while len(result) > k:
      heapq.heappop(result);
    result = [ heapq.heappop(result) for _ in range(len(result)) ];
    result = list(reversed(result));
    return result;

  def knn_rtree(self, q, k):
    q = self.pca.transform(np.array([q]))[0];
    print(q)
    res = list(self.idx.nearest(coordinates=(*q, *q), num_results=int(k)));
    return [ path[1:-1] for path in self.encodings.iloc[res, 0] ];