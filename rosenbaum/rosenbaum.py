import numpy as np
from math import comb, factorial, pow, log, exp 
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
import anndata as ad
import pandas as pd

from scipy.stats import rankdata
from scipy.spatial.distance import cdist as cpu_cdist
from scipy.sparse import csr_matrix

from itertools import chain
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


def calculate_distances(samples, metric):
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
    #
    num_samples = len(samples)

    if num_samples % 2 != 0: # with an uneven number of samples, a minimal-distance column is added 
        distances = np.pad(distances, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        num_samples += 1

    max_distance = np.max(distances)
    distances = max_distance + 1 - distances
    return distances


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
    print("creating distance graph with kNN.")
    sc.pp.neighbors(adata, n_neighbors=k, metric=metric)
    connectivities = adata.obsp['connectivities']
    
    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    G = gt.Graph(directed=False)
    num_nodes = connectivities.shape[0]
    G.add_vertex(num_nodes)

    weights = G.new_edge_property("float") 

    rows, cols = connectivities.nonzero()
    for row, col in zip(rows, cols):
        edge = G.add_edge(row, col)  
        weights[edge] = connectivities[row, col]

    G.edge_properties["weight"] = weights
    return G, weights

            
def match_samples(adata, metric, k):
    print("matching samples.")
    num_samples = len(adata)
    if k:
        G, weights = construct_graph_via_kNN(adata, metric, k)
    else:
        distances = calculate_distances(adata.X, metric)
        G, weights = construct_graph_from_distances(distances)
        
    matching = max_cardinality_matching(G, weight=weights, minimize=False) # "minimize=True" only works with a heuristic, therefore we use (max_distance + 1 - distance_ij) and maximize 
    matching_list = extract_matching(matching)
    matching_list = [p for p in matching_list if ((p[0] < num_samples) and (p[1] < num_samples))] # TODO: check if this fully filters invalid matches!!!
    return matching_list


def cross_match_count(Z, matching, test_group):
    print("counting cross matches")
    pairs = [(Z.iloc[i], Z.iloc[j]) for (i, j) in matching]
    filtered_pairs = [pair for pair in pairs if (pair[0] == test_group) ^ (pair[1] == test_group)] # cross-match pairs contain test group exactly once
    a1 = len(filtered_pairs)
    return a1



def get_p_value(a1, n, N, I):
    p_value = 0
    for A1 in range(a1 + 1):  # For all A1 <= a1
        if (n - A1) % 2 != 0: # A2 needs to be an integer
            continue

        if (I % 2) != (((n + A1) / 2) % 2): # A0 needs to be an integer
            continue

        A2 = (n - A1) / 2 
        A0 = I - (n + A1) / 2 # Remaining pairs are B-B

        if A0 < 0 or A2 < 0: # invalid
            continue  

        log_numerator = A1 * log(2) + log(factorial(I))
        log_denominator = log(comb(N, n)) + log(factorial(A0)) + log(factorial(A1)) + log(factorial(A2))
        
        p_value += exp(log_numerator - log_denominator)
    
    return p_value
    
        
def get_z_score(a1, n, N):
    m = N - n
    E = n * m / (N - 1) # Eq. 3 in Rosenbaum paper
    var = 2 * n * (n - 1) * m * (m - 1) / ((N - 3) * (N - 1)**2)
    z = (a1 - E) / np.sqrt(var)
    return z


def get_relative_support(N, Z):
    max_support = len(Z) - (len(Z) % 2)
    return N / max_support
    

def rosenbaum_test(Z, matching, test_group):
    used_elements = list(chain.from_iterable(matching))
    n = sum(1 for el in used_elements if Z.iloc[el] == test_group)
    N = len(matching) * 2
    I = len(matching)

    a1 = cross_match_count(Z, matching, test_group)
    
    p_value = get_p_value(a1, n, N, I)
    z_score = get_z_score(a1, n, N)
    relative_support = get_relative_support(N, Z)
    return p_value, z_score, relative_support


def rosenbaum(adata, group_by, test_group, reference="rest", metric="mahalanobis", rank=False, k=None):
    """
    Perform Rosenbaum's matching-based test for checking the association between two groups 
    using a distance-based matching approach.

    Parameters:
    -----------
    data : anndata.AnnData or pd.DataFrame
        The input data containing the samples and their respective features. If the input is an
        `AnnData` object, the samples and their corresponding features should be stored in `data.X` and the
        group labels in `data.obs[group_by]`. If using a `pandas.DataFrame`, the group labels should be in the
        column specified by `group_by`, and the feature matrix should be the remaining columns.

    group_by : str
        The column in `data.obs` or `data` (in case of a `pandas.DataFrame`) containing the group labels.
        The values of this column should include the `test_group` and potentially the `reference` group.

    test_group : str
        The group of interest that is being tested for association. This group will be compared against the `reference` group.

    reference : str, optional, default="rest"
        The group used as a comparison to the `test_group`. If set to "rest", all groups other than `test_group`
        are treated as the reference group.

    metric : str, optional, default="mahalanobis"
        The distance metric used for calculating distances between the samples during the matching process. 
        It can be any valid metric recognized by `scipy.spatial.distance.cdist`.

    rank : bool, optional, default=True
        If `True`, ranks the features in the data matrix before performing the matching. This can help reduce
        the impact of varying scales of the features on the distance computation.

    Returns:
    --------
    p_value : float
        The p-value from Rosenbaum's test, indicating the statistical significance of the observed matching.

    a1 : int
        The count of cross-matched pairs that contain `test_group` exactly once. This is used to compute the p-value.

    Raises:
    -------
    TypeError : If the input `data` is neither an `AnnData` object nor a `pandas.DataFrame`.
    ValueError : If the input `test_group` is not in the data.

    Notes:
    ------
    Rosenbaum's test compares how likely it is to observe a matching between the `test_group` and the `reference`
    group, using a matching algorithm based on distance metrics (such as "mahalanobis"). The test computes a p-value
    based on the number of cross-matched pairs between the two groups.

    The function internally uses the `match_samples` function to compute a matching of the samples based on the chosen
    distance metric. The resulting matching is then used in the `rosenbaum_test` to calculate the p-value.
    """

    if not isinstance(adata, ad.AnnData):
        raise TypeError("the input must be an AnnData object or a pandas DataFrame.")

    if test_group not in adata.obs[group_by].values:
        raise ValueError("the test group is not contained in your data.")
    
    if rank:
        print("computing variable-wise ranks.")
        adata.X = np.apply_along_axis(rankdata, axis=0, arr=adata.X)

    if reference != "rest":
        mask = (adata.obs[group_by].values == test_group) | (adata.obs[group_by].values == reference)
        group_data = adata.obs[group_by]
        adata = ad.AnnData(adata.X[mask, :])
        adata.obs[group_by] = group_data[mask].values
        print("filtered samples.")

    matching = match_samples(adata, metric=metric, k=k)
    return rosenbaum_test(Z=adata.obs[group_by], matching=matching, test_group=test_group)