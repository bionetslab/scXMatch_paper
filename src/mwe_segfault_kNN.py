import numpy as np
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
import anndata as ad
import scanpy as sc 
from scipy.sparse import csr_matrix, tril, triu

np.random.seed(42)

samples = np.random.normal(0, 1, [10000, 2])
adata = ad.AnnData(samples)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_neighbors=5, metric="sqeuclidean")

distances = adata.obsp["distances"].copy()
max_dist = distances.max() 
distances.data = max_dist + 1 - distances.data # transform so that weight minimization ~ weight maximization

lower_tri = tril(distances)
upper_tri = triu(distances)

transposed_lower_tri = lower_tri.transpose()
sparse_distances = transposed_lower_tri.maximum(upper_tri)


G = gt.Graph(sparse_distances, directed=False) 
    
print(len(G.get_edges()), "edges", len(G.get_vertices()), "nodes")

matching = max_cardinality_matching(G, weight=G.edge_properties["weight"], minimize=False, brute_force=True)
