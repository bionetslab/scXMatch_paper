import numpy as np
import sys
sys.path.append("..")
from src import *
import itertools
import logging
import time
import concurrent.futures


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
    if k:
        G = construct_graph_via_kNN(adata)
    else:
        distances = calculate_distances(adata.X.toarray(), metric)
        G = construct_graph_from_distances(distances)
    matching = match(G)
    #matching = [sorted(m) for m in matching]
    return matching 
    

def run_test(k, n_obs, n_var, metric):
    adata = simulate_data(n_obs, n_var)
    
    t0 = time.time()
    if k:
        if k < n_obs:
            kNN(adata, k, metric)
            t1 = time.time()
            
            matching_nx = test_nx(adata, k, metric)
            t2 = time.time()
            logging.info(f"{k}; {n_obs}; {n_var}; {t1 - t0:.6f}; {t2 - t1:.6f}; -1")
            
            matching_gt = test_gt(adata, k, metric)
            t3 = time.time()

            # Sort matchings for comparison
            matching_nx = sorted([sorted(m) for m in matching_nx], key=lambda m: m[0])

            matching_gt = sorted([sorted(m) for m in matching_gt], key=lambda m: m[0])
            
            # Check if matchings are equal
            match_check = matching_nx == matching_gt

            # Log the results directly in the function
            logging.info(f"{k}; {n_obs}; {n_var}; {t1 - t0:.6f}; {t2 - t1:.6f}; {t3 - t2:.6f}")
            return k, n_obs, n_var, t1 - t0, t2 - t1, t3 - t2, match_check
        else:
            print("skipped", k, n_obs)
            return
    return


def main():
    logging.basicConfig(
        filename="/data/bionets/je30bery/rosenbaum_test/evaluation_results/runtime/runtime_log.txt",
        level=logging.INFO,
        format="%(message)s"
    )

    metric = "sqeuclidean"
    k_values = [2, 5, 10, None]
    n_obs_values = [100, 1000, 5000]
    n_var_values = [100, 1000, 2000, 5000]
    parameter_combinations = [(10, 5000, 1000), 
                              (10, 5000, 5000)]
    # Prepare logging header
    logging.info(f"k; n_obs; n_var; t[s] PCA; t[s] NX; t[s] GT")

    # Start measuring the overall runtime

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for k, n_obs, n_var in parameter_combinations:
            futures.append(executor.submit(run_test, k, n_obs, n_var, metric))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Ensure that the result is fetched and logged, even in case of error
            except Exception as e:
                logging.error(f"Error in a future task: {e}")




if __name__ == "__main__":
    main()