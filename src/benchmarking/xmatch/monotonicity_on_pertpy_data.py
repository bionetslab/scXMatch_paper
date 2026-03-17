import anndata as ad
import os
import pandas as pd
from scxmatch import test
import sys
import time


def get_xm_log(adata, group_by, reference, k):
    results_list = []
    for test_group in adata.obs[group_by].unique():
        if test_group == reference:
            continue
        start_time = time.time()
        p, z, s = test(adata, group_by=group_by, reference=reference, test_group=test_group, rank=False, metric="sqeuclidean", k=k)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results_list.append({
            "testgroup": test_group,
            "reference": reference,
            "relative_support": s,
            "time_test": elapsed_time,
            "P": p,
            "k": k
        })

    results_df = pd.DataFrame(results_list)
    return results_df


def main(dataset_path):
    name = sys.argv[1]
    names = sorted([f for f in os.listdir(dataset_path) if (f.endswith("hdf5") and (name in f))])
    files = [os.path.join(dataset_path, f) for f in names]
    print("DATASET NAME", name)
    
    
    for f in files:
        basen = os.path.basename(f)
        p = f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_5_effect_k/different_k_xm_benchmark_results_{basen}.csv"

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
        
        result_dfs = list()
        k_max = len(adata) - 1
        for k in [10, 20, 50, 100, 500, 1000, 2000, "full"]:
            if k != "full":
                if k < k_max:
                    results_df = get_xm_log(adata, group_by, reference, k)
                    result_dfs.append(results_df)
            else:
                results_df = get_xm_log(adata, group_by, reference, k)
                result_dfs.append(results_df)
                
        results_df = pd.concat(result_dfs)
        results_df.to_csv(p, index=False)

        

if __name__ == "__main__":
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/")