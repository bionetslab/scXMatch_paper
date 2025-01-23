import networkx as nx
from scipy.sparse import csr_matrix
import scanpy as sc

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
    try:
        if GPU:
            print("trying to use GPU to calculate distance matrix.")
        distances = cp.asnumpy(gpu_cdist(cp.array(samples), cp.array(samples), metric=metric)) 

    except:
        if GPU:
            print("using CPU to calculate distance matrix due to chosen metric.")
        else:
            print("using CPU to calculate distance matrix.")
        distances = cpu_cdist(samples, samples, metric=metric)
    return distances


def construct_graph_from_distances_nx(distances):
    print("creating distance graph.")
    G = nx.from_numpy_array(distances)
    del distances
    return G


def construct_graph_via_kNN_nx(adata, metric, k):
    print("calculating PCA and kNN graph.")
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=k, metric=metric)

    print("extracting connectivities.")
    connectivities = adata.obsp['connectivities']
    del adata
    print(connectivities.toarray())

    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    print("assembling edges")
    G = nx.from_scipy_sparse_array(connectivities)
    del connectivities
    return G


def match_nx(G):
    matching = nx.min_weight_matching(G)     
    del G
    return matching
            

