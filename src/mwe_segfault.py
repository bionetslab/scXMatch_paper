import numpy as np
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
from scipy.spatial.distance import cdist 
from scipy.sparse import csr_matrix

p = 0.4
distances = np.random.normal(0, 1, [1000, 1000])
distances *= (distances > p)
distances = np.tril(distances, k=-1).transpose() 

sparse_distances = csr_matrix(distances)
print(sparse_distances.shape)
G = gt.Graph(sparse_distances, directed=False)
print(len(G.edge_properties["weight"]), "edges")
matching = max_cardinality_matching(G, weight=G.edge_properties["weight"], minimize=False) # SEGFAULT
print(matching.get_array())
print("matching done")