import pertpy as pt
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
import scanpy as sc
import time

def get_e_distance_log(adata, group_by, reference, subsampling):
    group_counts = adata.obs[group_by].value_counts()
    if subsampling:
        print("WARNING: subsampling enabled, results will be based on a subset of the data.", flush=True, file=sys.stderr)
    # --- Balance groups ---
        min_count = group_counts.min()

        print(f"Minimum group size: {min_count}", flush=True, file=sys.stderr)

        sampled_indices = []
        relative_support_dict = dict()

        for g in group_counts.index:
            idx = np.where(adata.obs[group_by] == g)[0]
            sampled = np.random.choice(idx, min_count, replace=False)
            sampled_indices.append(sampled)
            relative_support_dict[g] = len(sampled) / len(idx)

        sampled_indices = np.concatenate(sampled_indices)
        subset = adata[sampled_indices].copy()
    else:
        relative_support_dict = dict()
        for g in group_counts.index:
            relative_support_dict[g] = 1
        subset = adata.copy()
    
    # --- PCA ---
    n_comps = min(50, subset.shape[1] - 1, subset.shape[0] - 1)
    t1 = time.time()
    sc.pp.pca(subset, n_comps=n_comps)
    t2 = time.time()
    time_pca = t2 - t1
    print(f"PCA took {time_pca:.2f} seconds", flush=True, file=sys.stderr)

    # --- Distance test ---
    
    rows = list()
    for group in group_counts.index:
        if group == reference:
            continue
        t3 = time.time()
        Distance = pt.tools.Distance(metric="edistance")

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
    results_df = pd.DataFrame(rows)
    return results_df


def main(dataset_path):
    names = sorted([f for f in os.listdir(dataset_path) if f.endswith("hdf5")])
    files = [os.path.join(dataset_path, f) for f in names]
    subsampling = False
    print("files", files)
    
    for f in files:
        if subsampling:
            p = os.path.basename(f).split(".")[0] + "_balanced"
        else:
            p = os.path.basename(f).split(".")[0] + "_unbalanced"
        p = os.path.join("/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_14_var_edistance", f"{p}_e_distance_results.csv")
        
        adata = ad.read_h5ad(f)
        if "mcfarland" in f:
            group_by = "pert_time"
            reference = "control"
            
        elif "norman" in f:
            group_by = "n_guides"
            reference = "control"
            
        elif "schiebinger" in f:
            group_by = "perturbation"
            reference = "control"
            
        elif "bhatta" in f:
            group_by = "label"
            reference = "Maintenance_Cocaine"
        
        elif "Mimitou" in f:
            group_by = "perturbation"
            reference = "control"
            
        else:
            raise ValueError("Unknown dataset")
        
        adata = ad.read_h5ad(f)
        adata = adata[adata.obs[group_by].notna()].copy()
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        
        groups =  adata.obs[group_by].unique()
        all_results = []
        for test_group in groups:
            if test_group == reference:
                continue
            for group_by_split in ["split_10", "split_30", "split_50"]:
                subset_1 = adata[( (adata.obs[group_by] == test_group) & (adata.obs[group_by_split] == 1) ), :].obs.index
                for group_by_split_reference in ["split_10", "split_30", "split_50"]:    
                    subset_2 = adata[( (adata.obs[group_by] == reference) & (adata.obs[group_by_split_reference] == 1) ), :].obs.index
                    results = get_e_distance_log(adata[list(subset_1) + list(subset_2)], group_by=group_by, reference=reference, test_group=test_group)   
                    results["group_by_split"] = group_by_split
                    results["group_by_split_reference"] = group_by_split_reference
                    results["test_group"] = test_group
                    all_results.append(results)
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(p, index=False)
        
if __name__ == "__main__":
    print("hello")
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji")