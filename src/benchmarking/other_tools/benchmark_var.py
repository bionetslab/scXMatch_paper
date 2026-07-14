import anndata as ad
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_config import get_config
from benchmark_monotonicity import *

def main(dataset_path="/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/"):
    dataset_name = sys.argv[1]
    names = [f for f in os.listdir(dataset_path) if (f.endswith("h5ad") and (dataset_name in f))]
    files = [os.path.join(dataset_path, f) for f in names]

    os.makedirs(f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_3_var_benchmark/", exist_ok=True)
    
    for f in files:
        adata = ad.read_h5ad(f)
        group_by, reference = get_config(f)
        
        print("reading", f)
        adata = ad.read_h5ad(f)
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        
        groups = sorted(adata.obs[group_by].unique())[::-1]
        
        
        for test_group in groups:
            if test_group == reference:
                continue
            for group_by_split in ["split_10", "split_30", "split_50"]:
                subset_1 = adata[( (adata.obs[group_by] == test_group) & (adata.obs[group_by_split] == 1) ), :].obs.index
                for group_by_split_reference in ["split_10", "split_30", "split_50"]:
                    if os.path.exists(f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_3_var_benchmark/edist_benchmark_results_{os.path.basename(f)}_{test_group}_{group_by_split}_ref_{group_by_split_reference}.csv"):
                        print("skipi")
                        continue
                    subset_2 = adata[( (adata.obs[group_by] == reference) & (adata.obs[group_by_split_reference] == 1) ), :].obs.index
                    adata_subset = adata[list(subset_1) + list(subset_2)].copy()
                    print(adata_subset)
                    print(adata_subset.obs[[group_by, 'pseudo_bulk_100']].value_counts())
                    results = benchmark_all(adata_subset, group_by, reference=reference)
                    print(results)
                    results_df = pd.DataFrame(results)   
                    results_df.to_csv(f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_3_var_benchmark/edist_benchmark_results_{os.path.basename(f)}_{test_group}_{group_by_split}_ref_{group_by_split_reference}.csv", index=True)

                
if __name__ == "__main__":
    main()