import numpy as np
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
import anndata as ad
import scanpy as sc 
from scipy.sparse import csr_matrix

np.random.seed(42)

samples = np.random.normal(0, 1, [2000, 2])
print(samples.shape) # (2000, 2)
adata = ad.AnnData(samples)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_neighbors=5, metric="sqeuclidean")

distances = adata.obsp["distances"].copy()
del adata

max_dist = distances.max() 
distances.data = max_dist + 1 - distances.data
transposed_distances = distances.transpose()
combined_distances = np.maximum(distances.todense(), transposed_distances.todense())
np.fill_diagonal(combined_distances, 0)

sparse_weights = csr_matrix(combined_distances)


G = gt.Graph(distances, directed=False) 
print(f"Self-loops: {sum(1 for e in G.edges() if e.source() == e.target())}") # Self-loops: 0
print(f"Parallel edges: {len(set(G.edges())) != G.num_edges()}") # Parallel edges: False
print(len(G.edge_properties["weight"])) # 8020

matching = max_cardinality_matching(G, weight=G.edge_properties["weight"], minimize=False)
