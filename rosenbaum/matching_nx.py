import numpy as np
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
import networkx as nx
from tqdm import tqdm
from scipy.spatial.distance import cdist as cpu_cdist
from scipy.sparse import csr_matrix
import scanpy as sc


try:
    from cupyx.scipy.spatial.distance import cdist as gpu_cdist
    import cupy as cp
    GPU = True
    print("found cupy installation, will try use the GPU to calculate the distance matrix.")
except:
    GPU = False
    print("will use the CPU to calculate the distance matrix.")
    pass


def construct_graph_from_distances(distances):
    print("creating distance graph.")
    G = nx.from_numpy_array(distances)
    return G, distances



def construct_graph_via_kNN(adata, metric, k):
    # is it better to calculate the distances first, and then calculate the NNs on that
    # or to calculate the NNs and then calculate the distances only for these 
    print("calculating PCA and kNN graph.")

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=k, metric=metric)

    print("extracting connectivities.")
    connectivities = adata.obsp['connectivities']
    del adata
    print(connectivities.shape[0])

    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    print("assembling edges")
    G = nx.Graph(connectivities)
    del connectivities
    return G


def match(G, weights, num_samples):
    matching = nx.min_weight_matching(G)     
    del G
    return matching
            

