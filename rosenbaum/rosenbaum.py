import numpy as np
from math import comb, factorial, pow
import anndata as ad
from scipy.stats import rankdata
from itertools import chain
from .matching import *
from .matching_nx import *

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
        A2 = (n - A1) / 2 
        A0 = I - (n + A1) / 2 

        if int(A0) != A0:
            continue
        if int(A2) != A2:
            continue 
        if A0 < 0 or A2 < 0: # invalid
            continue  

        #print("accepted")
        numerator = pow(2, A1) * factorial(I)
        denominator = comb(N, n) * factorial(A0) * factorial(A1) * factorial(A2)
        p_value += numerator / denominator

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


def rosenbaum(adata, group_by, test_group, reference="rest", metric="mahalanobis", rank=False, k=None, nx=False):
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
        adata = adata[adata.obs[group_by].isin([test_group, reference]), :]
        print("Filtered and downsampled samples:")
        print(adata.obs[group_by].value_counts())

    
    print("matching samples.")
    if nx: # NX based computation
        if k:
            G = construct_graph_via_kNN_nx(adata, metric, k)
        else:
            distances = calculate_distances_nx(adata.X, metric)
            G = construct_graph_from_distances_nx(distances)
        matching = match_nx(G)
        matching = [sorted(m) for m in matching]

    else: # graphtool based computation
        num_samples = len(adata)
        if k:
            G, weights = construct_graph_via_kNN(adata, metric, k)
        else:
            distances = calculate_distances(adata.X.toarray(), metric)
            G, weights = construct_graph_from_distances(distances)
        matching = match(G, weights, num_samples)
        matching = [sorted(m) for m in matching]


    return rosenbaum_test(Z=adata.obs[group_by], matching=matching, test_group=test_group), matching, G