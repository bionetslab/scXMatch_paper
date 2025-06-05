import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
from preprocess import scanpy_setup, assign_pseudo_bulks_and_splits

def pre_process_all(dataset_path):
    names = [f for f in os.listdir(dataset_path) if (not f.startswith("_") and os.path.isdir(os.path.join(dataset_path, f)))]
    files = [os.path.join(dataset_path, f, "gene_scores") for f in names]
    print(names)
    for n, f in tqdm(zip(names, files)):
        if ("Liscovitch" in n) or ("Mimitou" in n):

            print("NAME", n)
            mtx = sc.read_mtx(os.path.join(f, "counts.mtx.gz"))
            obs = pd.read_csv(os.path.join(f, "obs.csv"), sep=",", header=0)
            var = pd.read_csv(os.path.join(f, "var.csv"), sep=",", header=0)

            adata = ad.AnnData(X=mtx)
            adata.obs = obs
            adata.var = var
            
            adata = scanpy_setup(adata)
            
            group_by = "perturbation"
            
            adata = assign_pseudo_bulks_and_splits(adata, group_by)
            adata.write_h5ad(f"{dataset_path}/LSI_{n}.hdf5")
            print(adata)
        
if __name__ == "__main__":
    
    pre_process_all("/home/woody/iwbn/iwbn007h/data/scATACseq/")