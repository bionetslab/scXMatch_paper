import anndata as ad
from time import time
import pandas as pd
import scxmatch as xm
import glob
import os
import fcntl
import sys
import resource
from itertools import product
from runtime_on_splatter import get_peak_ram_gb, run_matching, write_row, process_file


splatter_dir = "/home/woody/iwbn/iwbn007h/xm_gaussian_data/random_normal_sims"
out_file = "/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/0_1_splatter_runtime/random_results_with_s.csv"


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(splatter_dir, "*.h5ad")))
    ks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    combinations = list(product(files, ks)) 

    idx = int(sys.argv[1])
    file, k = combinations[idx]
    print("FILE:", file, file=sys.stderr, flush=True)
    print("K:", k, file=sys.stderr, flush=True)  
    process_file(file, k, idx, out_file)