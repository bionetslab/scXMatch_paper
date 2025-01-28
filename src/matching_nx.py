import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np


try:
    from cupyx.scipy.spatial.distance import cdist as gpu_cdist
    import cupy as cp
    GPU = True
    print("found cupy installation, will try use the GPU to calculate the distance matrix.")
except:
    from scipy.spatial.distance import cdist as cpu_cdist
    GPU = False
    print("will use the CPU to calculate the distance matrix.")
    pass



def calculate_distances_nx(samples, metric):
    if not isinstance(samples, np.ndarray):  # Check if it's a scipy sparse matrix
        samples = samples.toarray()

    if GPU:
        try:
            print("trying to use GPU to calculate distance matrix.")
            distances = cp.asnumpy(gpu_cdist(cp.array(samples), cp.array(samples), metric=metric)) 

        except:
            print("using CPU to calculate distance matrix due to chosen metric.")
            distances = cpu_cdist(samples, samples, metric=metric)

    else:
        print("using CPU to calculate distance matrix.")

        distances = cpu_cdist(XA=samples, XB=samples, metric=metric)

    return distances


def construct_graph_from_distances_nx(distances):
    print("creating distance graph.")
    G = nx.from_numpy_array(distances)
    del distances
    return G


def construct_graph_via_kNN_nx(adata):
    #print("extracting connectivities.")
    distances = adata.obsp['distances']
    del adata

    if not isinstance(distances, csr_matrix):
        distances = csr_matrix(distances)

    #print("assembling edges")
    G = nx.from_scipy_sparse_array(distances)
    del distances
    return G


def match_nx(G):
    matching = nx.min_weight_matching(G)     
    del G
    return matching
            

