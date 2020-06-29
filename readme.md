## Integrantes
- Indhira Ramirez
- Reynaldo Rojas

## Librerias usadas
- pandas
- numpy
- face_recognition
- rtree
- sklearn.decomposition
- heapq
# Algoritmos de busqueda
## distacia euclidiana y manhattan
~~~

~~~

## KnnSearch con heap
~~~
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
~~~

## rtree

# Experimentacion
## Precision de distancia euclidiana vs manhattan

## Tiempos de Knn-RTree vs Knn-Secuencial
