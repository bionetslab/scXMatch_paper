import numpy as np
import itertools
import logging
import time
import sys
import anndata as ad
sys.path.append("..")
from src import kNN, construct_graph_via_kNN, calculate_distances, construct_graph_from_distances, match

# Simulate data with the same function
def simulate_data(n_obs, n_var):
    samples = [np.random.normal(0, 1, n_var) for _ in range(n_obs)]
    adata = ad.AnnData(np.array(samples))
    print(adata)
    return adata

# Test ground truth function
def test_gt(adata, k, metric):
    if k:
        G = construct_graph_via_kNN(adata)
    else:
        distances = calculate_distances(adata.X.toarray(), metric)
        G = construct_graph_from_distances(distances)
    matching = match(G, len(adata))
    return matching 


# Run the benchmarking test
def run_test(k, n_obs, n_var_values, metric):
    for n_var in n_var_values:
        print(n_var)
        adata = simulate_data(n_obs, n_var)
        
        if k:
            if k < n_obs:
                t0 = time.time()
                kNN(adata, k, metric)
                t1 = time.time()
                if n_var != sorted(n_var_values)[0]:
                    logging.info(f"{k}; {n_obs}; {n_var}; {t1 - t0:.6f}; -1")
                    continue
                matching_gt = test_gt(adata, k, metric)
                t2 = time.time()
                logging.info(f"{k}; {n_obs}; {n_var}; {t1 - t0:.6f}; {t2 - t1:.6f}")
                
            else:
                print(f"skipped {k} {n_obs}")
        else:
            t1 = time.time()
            matching_gt = test_gt(adata, k, metric)
            t2 = time.time()
            logging.info(f"None; {n_obs}; {n_var}; 0; {t2 - t1:.6f}")



def main():
    k = int(sys.argv[1]) # [10000, 7500, 5000, 2500, 1000, 500, 100]
    logging.basicConfig(
        filename=f"{k}_runtime_log.txt",
        level=logging.INFO,
        format="%(message)s"
    )
    logging.info(f"k; n_obs; n_var; t_NN [s]; t_matching [s]")

    print(k)
    metric = "sqeuclidean"
    
    if k == 2500:
        n_obs_values = [100000]
    else:
        n_obs_values = [50000, 100000]
    
    n_var_values =  [2]
    n_obs_values =[100]


    for n_obs in n_obs_values:
        print(n_obs)
        run_test(k, n_obs, n_var_values, metric)


if __name__ == "__main__":
    main()
