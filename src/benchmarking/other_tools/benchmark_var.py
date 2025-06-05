import pertpy as pt
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
from benchmark_monotonicity import * 


def main(dataset_path="/home/woody/iwbn/iwbn007h/data/scrnaseq_ji"):
    dataset_name = sys.argv[1]
    names = [f for f in os.listdir(dataset_path) if (f.endswith("hdf5") and f.startswith("processed") and (dataset_name in f))]
    files = [os.path.join(dataset_path, f) for f in names]
    print(names)
    print(files)
    
    for f in files:
        adata = ad.read_h5ad(f)
        if "mcfarland" in f:
            group_by = "perturbation_grouped"
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
        
        print("reading", f)
        adata = ad.read_h5ad(f)
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        
        groups = adata.obs[group_by].unique()
        
        for test_group in groups:
            if test_group == reference:
                continue
            for group_by_split in ["split_10", "split_30", "split_50", "split_1"]:
                subset_1 = adata[( (adata.obs[group_by] == test_group) & (adata.obs[group_by_split] == 1) ), :].obs.index
                for group_by_split_reference in ["split_10", "split_30", "split_50", "split_1"]:
                    try:
                        subset_2 = adata[( (adata.obs[group_by] == reference) & (adata.obs[group_by_split_reference] == 1) ), :].obs.index
                        print(adata[list(subset_1) + list(subset_2)].obs[group_by].value_counts())
                        results = benchmark_all(adata[list(subset_1) + list(subset_2)], group_by, reference=reference)
                        results_df = pd.DataFrame(results)
                        results_df.to_csv(f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_3_var_benchmark/benchmark_results_{os.path.basename(f)}_{test_group}_{group_by_split}_ref_{group_by_split_reference}.csv", index=True)
                    except:
                        continue
                
if __name__ == "__main__":
    main()