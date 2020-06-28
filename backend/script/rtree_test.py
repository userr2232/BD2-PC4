#sudo apt install python3-rtree
#http://toblerity.org/rtree/class.html

from rtree import index
p = index.Property()
p.dimension = 2
p.buffering_capacity = 3
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('2d_index',properties=p)

#insertar puntos
idx.insert(0, (1,1,1,1))
idx.insert(1, (1,2,1,2))
idx.insert(2, (3,1,3,1))
idx.insert(3, (1,3,1,3))
idx.insert(4, (2,3,2,3))

#retornar elementos de la interseccion con el rectangulo 
q = (2, 0, 4, 3)
lres = [n for n in idx.intersection(q)]
print("Elementos en (2, 0) - (4, 3): ", lres)

#retornar los k=1 vecinos de (3,3)
q = (3, 3, 3, 3)
lres = list(idx.nearest(coordinates=q, num_results=1))
print("El vecino mas cercano de (3,3): ", lres)

#retornar los k=2 vecinos de (3,3)
lres = list(idx.nearest(coordinates=q, num_results=2))
print("Los dos vecinos mas cercano de (3,3): ", lres)
