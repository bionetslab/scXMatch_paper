import pertpy as pt
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
import scanpy as sc
import time

def get_e_distance_log(adata, group_by, reference):
    # --- Balance groups ---
    group_counts = adata.obs[group_by].value_counts()
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

    # --- PCA ---
    n_comps = min(50, subset.shape[1] - 1, subset.shape[0] - 1)
    t1 = time.time()
    sc.pp.pca(subset, n_comps=n_comps)
    t2 = time.time()
    time_pca = t2 - t1
    print(f"PCA took {time_pca:.2f} seconds", flush=True, file=sys.stderr)

    # --- Distance test ---
    distance_test = pt.tl.DistanceTest("edistance", n_perms=10000, obsm_key="X_pca")
    t3 = time.time()
    tab = distance_test(subset, groupby=group_by, contrast=reference)
    t4 = time.time()
    time_edist_test = t4 - t3
    print(f"Distance test took {time_edist_test:.2f} seconds", flush=True, file=sys.stderr)

    # --- Build results DataFrame ---
    results_list = []
    for g in tab.index:
        if g == reference:
            continue
        results_list.append({
            "testgroup": g,
            "reference": reference,
            "relative_support": relative_support_dict[g],
            "time_pca": time_pca,
            "time_edist_test": time_edist_test,
            "P": tab.loc[g, "pvalue"],
            "P_adj": tab.loc[g, "pvalue_adj"]
        })

    results_df = pd.DataFrame(results_list)
    return results_df


def main(dataset_path):
    name = sys.argv[1]
    names = sorted([f for f in os.listdir(dataset_path) if (f.endswith("hdf5") and (name in f))]) #  ]
    files = [os.path.join(dataset_path, f) for f in names]
    print("DATASET NAME", name)
    
    for f in files:
        basen = os.path.basename(f)
        p = f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_4_edist/edist_benchmark_results_{basen}.csv"
        if os.path.exists(p):
            print(f"not skipping {p}, results already exist.", flush=True, file=sys.stderr)
            #continue
        else:
            print(f"Processing {p}", file=sys.stderr)
            
        adata = ad.read_h5ad(f)
        if "mcfarland" in f:
            group_by = "pert_time"
            reference = "control"
            
        elif "norman" in f:
            group_by = "n_guides"
            reference = "control"
            
        elif "sciplex" in f:
            group_by = "dose_value"
            reference = "0.0"
            
        elif "schiebinger" in f:
            group_by = "perturbation"
            reference = "control"
            
        elif "bhatta" in f:
            group_by = "label"
            reference = "Maintenance_Cocaine"
            
        else:
            raise ValueError("Unknown dataset")
        
        adata = ad.read_h5ad(f)
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        
        results_df = get_e_distance_log(adata, group_by, reference)
        results_df.to_csv(p, index=False)

        
if __name__ == "__main__":
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/")