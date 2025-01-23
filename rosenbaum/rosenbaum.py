import numpy as np
from math import comb, factorial, pow, log, exp 
from graph_tool.topology import max_cardinality_matching
import graph_tool.all as gt
import networkx as nx
import anndata as ad
import pandas as pd
from tqdm import tqdm
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
    
    print("found", A1, "crossmatches and", A0, "+", A2, "iso-matches")
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


def rosenbaum(adata, group_by, test_group, reference="rest", metric="mahalanobis", rank=False, k=None, balance=False):
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
        print("Original group counts:")
        print(adata.obs[group_by].value_counts())

        # Create masks for the two groups
        reference_mask = adata.obs[group_by] == reference
        test_mask = adata.obs[group_by] == test_group

        # TODO only when balance=True
        # Get group sizes
        reference_count = reference_mask.sum()
        test_count = test_mask.sum()

        # Determine the smaller group size
        min_size = min(reference_count, test_count)

        if min_size > 0:
            # Downsample both groups to the same size
            sampled_reference = adata.obs[reference_mask].sample(n=min_size, random_state=42).index
            sampled_test = adata.obs[test_mask].sample(n=min_size, random_state=42).index

            # Combine sampled indices
            sampled_indices = sampled_reference.union(sampled_test)

            # Subset the AnnData object
            adata = adata[sampled_indices, :]

            print("Filtered and downsampled samples:")
            print(adata.obs[group_by].value_counts())
        else:
            print("One of the groups has no samples available after filtering.")

    
    print("matching samples.")
    num_samples = len(adata)
    if k:
        G, weights = construct_graph_via_kNN(adata, metric, k)
    else:
        distances = calculate_distances(adata.X.toarray(), metric)
        G, weights = construct_graph_from_distances(distances)
    
    matching = match(G, weights, num_samples)

    return rosenbaum_test(Z=adata.obs[group_by], matching=matching, test_group=test_group)