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
DE_l2 = lambda x,y : sum((x-y)**2)**0.5
DM_l1 = lambda x,y : sum(abs(x-y))
~~~
La distancia euclidiana, que mide la distancia de dos puntos, se mide como la logitud del segmento que los une y paraello usamos el teorema de pitagoras. Mientras que la manhattan mide la distancia como la suma de las diferencias de sus coordenadas. 

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
Se tiene data como todas las tuplas, Q como el vector caracteristico de la foto que buscamos y k como el numero maximo de resultados que buscaremos.
La busqueda knn con heap itera por toda la data y agrega esta a un max heap cordenado por -d, negativa de la distancia, respecto a el vector caracteristico Q. Se usa -d ya que lo que buscamos son los vectores caracteristicos con mayor similitud y por ende con menor distancia, al ser un max heap nos conviene agregarlo con -d. Despues de agregar todos los datos al heap, quitamos los elementos con menor cercania, o mayor distancia, hasta quedarnos con un heap de tamaño k. Por ultimo revertimos el orden de la lista para que los primeros en esta sean los mas cercanos al vector caracteristico Q.

## rtree
~~~
p = index.Property()
p.dimension = 32
p.buffering_capacity = 5
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index(properties=p)

pca = PCA(n_components=32);
~~~
Definimos p como nuestro indice y le ponemos dimencio 32, que es el numero de clusters que podra tener. Buffer_capacity es nuestro maximo de puntos por cada uno de estos clusters.
~~~
pca_data = pca.fit_transform(data);
k = 16
q = pca_data[5];
print("target", encodings.iloc[5,129]);
np.delete(pca_data, 0, 0);

for i in range(len(pca_data)):
  x = pca_data[i];
  idx.insert(i, (*x, *x));
~~~

# Experimentacion
## Precision de distancia euclidiana vs manhattan
![](fotos/p1.png)
En nuestra data no hubo mucha diferencia al usar ambas distancias.
## Tiempos de Knn-RTree vs Knn-Secuencial
