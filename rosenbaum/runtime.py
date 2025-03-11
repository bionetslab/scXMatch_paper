import numpy as np
from rosenbaum import *
import timeit
import itertools
import logging


logging.basicConfig(
    filename="runtime_gt_log.txt",
    level=logging.INFO,
    format="%(message)s"
)

def simulate_data(n_obs, n_var):
    samples = [np.random.normal(0, 1, n_var) for _ in range(n_obs)]
    adata = ad.AnnData(np.array(samples))
    return adata


def test_nx(adata, k, metric):
    if k:
        G = construct_graph_via_kNN_nx(adata)
    else:
        distances = calculate_distances_nx(adata.X, metric)
        G = construct_graph_from_distances_nx(distances)
    matching = match_nx(G)
    #matching = [sorted(m) for m in matching]
    return matching 


def test_gt(adata, k, metric):
    num_samples = len(adata)
    if k:
        G, weights = construct_graph_via_kNN(adata)
    else:
        distances = calculate_distances(adata.X.toarray(), metric)
        G, weights = construct_graph_from_distances(distances)
    matching = match(G, weights, num_samples)
    #matching = [sorted(m) for m in matching]
    return matching 
    

def main():

    #n_obs = 100
    #n_var = 2
    #k = 5
    reps = 10
    metric = "sqeuclidean"
    k_values = [2, 5, 10, 15]
    n_obs_values = [10, 100, 1000, 5000]
    n_var_values = [2] # [10, 100, 1000, 5000]
    parameter_combinations = itertools.product(k_values, n_obs_values, n_var_values)

    # Loop over parameter combinations            
    logging.info(f"k; n_obs; n_var; t[s]")

    for k, n_obs, n_var in parameter_combinations:
        try:

            # Generate data
            adata = simulate_data(n_obs, n_var)
            kNN(adata, k, metric)
                
            # Time `test_nx`
            #nx_time = timeit.timeit(lambda: test_nx(adata, k, metric), number=reps)
            #avg_nx_time = nx_time / reps
            #logging.info(f"{k}; {n_obs}; {n_var}; {avg_nx_time:.6f}")
            #print(f"{k}; {n_obs}; {n_var}; {avg_nx_time:.6f}")
            # Time `test_gt`
            gt_time = timeit.timeit(lambda: test_gt(adata, k, metric), number=reps)
            avg_gt_time = gt_time / reps
            print(f"{k}; {n_obs}; {n_var}; {avg_gt_time:.6f}")
            logging.info(f"{k}; {n_obs}; {n_var}; {avg_gt_time:.6f}")

        except Exception as e:
            logging.error(f"Error occurred with k={k}, n_obs={n_obs}, n_var={n_var}: {e}")
            continue  # Continue to the next combination even if one fails

    #return rosenbaum_test(Z=adata.obs[group_by], matching=matching, test_group=test_group), matching, G


if __name__ == "__main__":
    main()