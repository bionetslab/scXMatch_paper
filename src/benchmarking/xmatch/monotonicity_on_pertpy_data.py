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
        test_result = test(adata, group_by=group_by, reference=reference, test_group=test_group, rank=False, metric="sqeuclidean", k=k)
        end_time = time.time()
        elapsed_time = end_time - start_time
        test_result.update({"testgroup": test_group, "reference": reference, "k": k, "time_test": elapsed_time})
        
        results_list.append(test_result)
    results_df = pd.DataFrame(results_list)
    return results_df


def main(dataset_path):
    name = sys.argv[1]
    names = sorted([f for f in os.listdir(dataset_path) if (f.endswith("h5ad") and (name in f))])
    files = [os.path.join(dataset_path, f) for f in names]
    print("DATASET NAME", name)
    
    for f in files:
        basen = os.path.basename(f)
        p = f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_6_monotonicity_effect_size/results_{basen}_effect_size.csv"

        adata = ad.read_h5ad(f)
        if "mcfarland" in f:
            group_by = "pert_time"
            reference = "control"
            
        elif "norman_" in f:
            group_by = "label"
            reference = "0"
            
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
        
        results_df = get_xm_log(adata, group_by, reference, k=100)
        results_df.to_csv(p, index=False)

        

if __name__ == "__main__":
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji")