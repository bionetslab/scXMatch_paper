import numpy as np
import itertools
import logging
import time
import sys
import psutil
import anndata as ad
sys.path.append("../../scxmatch/src")
from scxmatch import kNN, construct_graph_via_kNN, calculate_distances, construct_graph_from_distances, match

# Simulate data with the same function
def simulate_data(n_obs, n_var):
    samples = [np.random.normal(0, 1, n_var) for _ in range(n_obs)]
    adata = ad.AnnData(np.array(samples))
    return adata

# Test ground truth function
def test_gt(adata, k, metric, distances=None):
    if k:
        G = construct_graph_via_kNN(adata)
    else:
        G = construct_graph_from_distances(distances)
    matching = match(G, len(adata))
    num_edges = len(G.get_edges())
    return matching, num_edges 

# Run the benchmarking test
def run_test(k, n_obs, n_var, metric):
    process = psutil.Process()
    start_mem = process.memory_info().rss  # Initial memory usage

    adata = simulate_data(n_obs, n_var)

    if k:
        if k < n_obs:
            t0 = time.time()
            kNN(adata, k, metric)
            t1 = time.time()
            peak_mem = process.memory_info().rss  # Memory after kNN
            matching_gt, n_edges = test_gt(adata, k, metric)
            t2 = time.time()
            peak_mem_matching = process.memory_info().rss  # Memory after matching  
            s = len(matching_gt) * 2 / len(adata)
            logging.info(f"{k},{n_obs},{n_var},{n_edges},{t1 - t0:.6f},{t2 - t1:.6f},{(peak_mem - start_mem) / (1024 * 1024):.2f} MB,{(peak_mem_matching - start_mem) / (1024 * 1024):.2f} MB,{s}")
            
        else:
            print(f"skipped {k} {n_obs}")
    else:
        t0 = time.time()
        distances = calculate_distances(adata.X, metric)
        t1 = time.time()
        peak_mem = process.memory_info().rss  # Memory after kNN
        matching_gt, n_edges  = test_gt(adata, k, metric, distances)
        t2 = time.time()
        peak_mem_matching = process.memory_info().rss  # Memory after matching
        logging.info(f"None,{n_obs},{n_var},{n_edges},{t1 - t0:.6f},{t2 - t1:.6f},{(peak_mem - start_mem) / (1024 * 1024):.2f} MB,{(peak_mem_matching - start_mem) / (1024 * 1024):.2f} MB")

def main():
    k = int(sys.argv[1])
    if k == 0:
        k = None
    n_obs = int(sys.argv[2])
    n_var = int(sys.argv[3])
    
    if k:
        if k >= n_obs:
            return

    print("k", k, "n_obs", n_obs, "n_var", n_var, file=sys.stderr)
    
    logging.basicConfig(
        filename=f"../evaluation_results/runtime_memory_log_{n_obs}.txt",
        level=logging.INFO,
        format="%(message)s"
    )
    logging.info(f"k,n_obs,n_var,n_edges,t_NN [s],t_matching [s],Peak Memory [MB] kNN,Peak Memory [MB] matching,s")
    metric = "sqeuclidean"
    run_test(k, n_obs, n_var, metric)


if __name__ == "__main__":
    main()