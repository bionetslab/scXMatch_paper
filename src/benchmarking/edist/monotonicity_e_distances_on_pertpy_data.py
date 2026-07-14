import pertpy as pt
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
import scanpy as sc
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_config import get_config

def get_e_distance_log(adata, group_by, reference, subsampling):
    group_counts = adata.obs[group_by].value_counts()
    if subsampling:
        print("WARNING: subsampling enabled, results will be based on a subset of the data.", flush=True, file=sys.stderr)
        min_count = group_counts.min()
        print(f"Minimum group size: {min_count}", flush=True, file=sys.stderr)

        sampled_indices = []
        relative_support_dict = {}
        for g in group_counts.index:
            idx = np.where(adata.obs[group_by] == g)[0]
            sampled = np.random.choice(idx, min_count, replace=False)
            sampled_indices.append(sampled)
            relative_support_dict[g] = len(sampled) / len(idx)

        subset = adata[np.concatenate(sampled_indices)].copy()
    else:
        relative_support_dict = {g: 1 for g in group_counts.index}
        subset = adata.copy()

    n_comps = min(50, subset.shape[1] - 1, subset.shape[0] - 1)
    t1 = time.time()
    sc.pp.pca(subset, n_comps=n_comps)
    t2 = time.time()
    time_pca = t2 - t1
    print(f"PCA took {time_pca:.2f} seconds", flush=True, file=sys.stderr)

    Distance = pt.tools.Distance(metric="edistance", obsm_key="X_pca")
    rows = []
    for group in group_counts.index:
        if group == reference:
            continue
        t3 = time.time()
        X = subset.obsm["X_pca"][subset.obs[group_by] == group]
        Y = subset.obsm["X_pca"][subset.obs[group_by] == reference]
        d = Distance(X, Y)
        t4 = time.time()
        rows.append({
            "testgroup": group,
            "reference": reference,
            "relative_support": relative_support_dict[group],
            "time_pca": time_pca,
            "time_edist_test": t4 - t3,
            "distance": d})
    return pd.DataFrame(rows)


def main(dataset_path):
    name = sys.argv[1]
    names = sorted([f for f in os.listdir(dataset_path) if (f.endswith("h5ad") and (name in f))])
    files = [os.path.join(dataset_path, f) for f in names]
    subsampling = False
    print("files", files)
    
    for f in files:
        if subsampling:
            p = os.path.basename(f).split(".")[0] + "_balanced"
        else:
            p = os.path.basename(f).split(".")[0] + "_unbalanced"
        p = os.path.join("/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_12_monotonicity_edistance", f"{p}_e_distance_results.csv")
        
        adata = ad.read_h5ad(f)
        group_by, reference = get_config(f)
        
        adata = ad.read_h5ad(f)
        adata = adata[adata.obs[group_by].notna()].copy()
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        
        results_df = get_e_distance_log(adata, group_by, reference, subsampling=subsampling)
        results_df.to_csv(p, index=False)

        
if __name__ == "__main__":
    print("hello")
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji")