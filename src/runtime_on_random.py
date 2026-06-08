import anndata as ad
from time import time
import pandas as pd
from scxmatch import test
import scxmatch as xm
import glob
import os
import fcntl
import sys
import resource
from itertools import product

splatter_dir = "/home/woody/iwbn/iwbn007h/xm_gaussian_data/random_normal_sims"
out_file = "/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/0_1_splatter_runtime/random_results.csv"

def get_peak_ram_gb():
    """
    Returns peak RAM usage in GiB.
    Linux ru_maxrss is in KB.
    Includes child processes.
    """
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    total_gb = (self_kb + child_kb) / 1024 / 1024
    return total_gb


def run_matching(adata, k, metric="sqeuclidean"):

    t0 = time()
    xm._match._kNN(adata, k, metric)
    t1 = time()
    G = xm._match._construct_graph_via_kNN(adata)
    matching = xm._match._match(G, len(adata))
    t2 = time()
    num_edges = len(G.get_edges())
    
    result = dict()
    result["t_NN [s]"] = t1 - t0
    result["t_matching [s]"] = t2 - t1
    result["peak_ram_gb"] = get_peak_ram_gb()
    result["num_edges"] = num_edges
    result["metric"] = metric
    return result


def write_row(result_dict):
    df = pd.DataFrame([result_dict])

    file_exists = os.path.isfile(out_file)

    with open(out_file, "a") as f:
        # lock file to avoid race conditions
        fcntl.flock(f, fcntl.LOCK_EX)
        df.to_csv(f, header=not file_exists, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)


def process_file(file, k, idx):
    adata = ad.read_h5ad(file)
    n_obs, n_var = adata.shape

    result = run_matching(adata, k, "sqeuclidean")

    result["n_obs"] = n_obs
    result["n_var"] = n_var
    result["file"] = os.path.basename(file)
    result["k"] = k
    result["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    result["slurm_array_idx"] =  idx
    write_row(result)


if __name__ == "__main__":

    files = sorted(glob.glob(os.path.join(splatter_dir, "*.h5ad")))
    ks = [10, 100, 5000] #[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    combinations = list(product(files, ks))
    
    idx = int(sys.argv[1])
    file, k = combinations[idx]
    print("FILE:", file, file=sys.stderr, flush=True)
    print("K:", k, file=sys.stderr, flush=True)  
    print("FILE:", file, file=sys.stdout, flush=True)
    print("K:", k, file=sys.stdout, flush=True)  
    process_file(file, k, idx)