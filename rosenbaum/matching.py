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




def extract_matching(matching_map):
    matching_list = []
    matched = set()  # To keep track of processed vertices

    for v in matching_map.get_array().nonzero()[0]:  # Only consider vertices with matches
        partner = matching_map[v]
        if partner != -1 and partner not in matched:
            matching_list.append((v, partner))
            matched.add(v)
            matched.add(partner)
    
    return matching_list



def construct_graph_from_distances(distances):
    num_samples = distances.shape[0]
    print("creating distance graph.")

    G = gt.Graph(directed=False)
    G.add_edge_list([(i, j) for i in range(num_samples) for j in range(i+1, num_samples)])
    
    weights = G.new_edge_property("double")
    for edge in G.edges():
        i, j = int(edge.source()), int(edge.target())
        weights[edge] = distances[i, j]
    G.edge_properties["weight"] = weights
    return G, weights




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
    G = gt.Graph(directed=False)
    num_nodes = connectivities.shape[0]

    G.add_vertex(num_nodes)

    weights = G.new_edge_property("float") 

    rows, cols = connectivities.nonzero()
    for row, col in tqdm(zip(rows, cols)):
        edge = G.add_edge(row, col)  
        weights[edge] = connectivities[row, col]

    G.edge_properties["weight"] = weights
    del connectivities
    return G, weights


def match(G, weights, num_samples):
    matching = max_cardinality_matching(G, weight=weights, minimize=False) # "minimize=True" only works with a heuristic, therefore we use (max_distance + 1 - distance_ij) and maximize 
    matching_list = extract_matching(matching)
    del G
    matching_list = [p for p in matching_list if ((p[0] < num_samples) and (p[1] < num_samples))] # TODO: check if this fully filters invalid matches!!!
    return matching_list
            
