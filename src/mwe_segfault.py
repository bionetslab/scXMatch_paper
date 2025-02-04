import numpy as np
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
from scipy.spatial.distance import cdist 
from scipy.sparse import csr_matrix

t = 3
distances = np.random.normal(0, 1, [5000, 5000])
distances *= (distances > t)
distances = np.triu(distances, k=1) 
sparse_distances = csr_matrix(distances)
G = gt.Graph(sparse_distances, directed=False)

print(len(G.get_edges()), "edges", len(G.get_vertices()), "nodes") #16837 edges 5000 nodes
assert sum(1 for e in G.edges() if e.source() == e.target()) == 0 # No self-loops
assert len(set(G.edges())) == G.num_edges() # No parallel edges

matching = max_cardinality_matching(G, weight=G.edge_properties["weight"], minimize=False) # SEGFAULT
print("matching done")