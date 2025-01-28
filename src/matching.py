import numpy as np
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
from scipy.spatial.distance import cdist as cpu_cdist
from scipy.sparse import csr_matrix


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


def calculate_distances(samples, metric):
    if not isinstance(samples, np.ndarray):  # Check if it's a scipy sparse matrix
        samples = samples.toarray()

    
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

    num_samples = len(samples)

    if num_samples % 2 != 0: # with an uneven number of samples, a minimal-distance column is added 
        distances = np.pad(distances, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        num_samples += 1

    max_distance = np.max(distances)
    distances = max_distance + 1 - distances
    return distances


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




def construct_graph_via_kNN(adata):
    distances = adata.obsp['distances']
    #del adata

    if not isinstance(distances, csr_matrix):
        distances = csr_matrix(distances)
    max_dist = np.max(distances)

    #print("assembling edges")
    G = gt.Graph(directed=False)
    
    #num_nodes = distances.shape[0]
    #G.add_vertex(num_nodes) # this causes a segfault

    weights = G.new_edge_property("float") 

    rows, cols = distances.nonzero()
    for row, col in zip(rows, cols):
        edge = G.add_edge(row, col)  
        weights[edge] = max_dist + 1 - distances[row, col]

    G.edge_properties["weight"] = weights
    # del distances
    return G, weights


def match(G, weights, num_samples):
    print("matching.")
    matching = max_cardinality_matching(G, weight=weights, minimize=False) # "minimize=True" only works with a heuristic, therefore we use (max_distance + 1 - distance_ij) and maximize 
    matching_list = extract_matching(matching)
    # del G
    matching_list = [p for p in matching_list if ((p[0] < num_samples) and (p[1] < num_samples))] # TODO: check if this fully filters invalid matches!!!
    return matching_list
            
