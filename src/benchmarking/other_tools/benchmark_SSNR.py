import anndata as ad
import pandas as pd
import os
import sys
from benchmark_monotonicity import * 

def main(dataset_path="/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/"):    
    dataset_name = sys.argv[1]
    #test_group = sys.argv[2]
    names = [f for f in os.listdir(dataset_path) if (f.endswith("h5ad") and (dataset_name in f))]
    files = [os.path.join(dataset_path, f) for f in names]
    print(names, flush=True, file=sys.stderr)
    print(files, flush=True, file=sys.stderr)
    #os.makedirs(f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_2_SSNR_benchmark/", exist_ok=True)
    
    for f in files:
        adata = ad.read_h5ad(f)
        if "mcfarland" in f:
            group_by = "pert_time"
            reference = "control"
            
        elif "norman" in f:
            group_by = "label"
            reference = "0"
            
        elif "sciplex" in f:
            group_by = "dose_value"
            reference = "0.0"
            
        elif "schiebinger" in f:
            group_by = "perturbation"
            reference = "control"            
            
        elif "bhatta" in f:
            group_by = "ori_label"
            adata.obs.rename({"label": "ori_label"}, axis=1, inplace=True)
            print(adata.obs["ori_label"].value_counts())
            reference = "Maintenance_Cocaine"
            
            
        else:
            raise ValueError("Unknown dataset")
        
        print("reading", f, flush=True, file=sys.stderr)
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        
        groups = adata.obs[group_by].unique()
        for test_group in groups:
            if test_group == reference:
                continue

            subset = adata[adata.obs[group_by].isin([test_group]), :].copy()
            for group_by_split in ["split_10", "split_30", "split_50"]:
                p = f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_2_SSNR_benchmark/benchmark_results_{os.path.basename(f)}_{test_group}_{group_by_split}.csv"
                if os.path.exists(p):
                    continue
                subset.obs[group_by_split] = subset.obs[group_by_split].astype(str)
                results = benchmark_all(subset, group_by_split, reference="0.0")
                results_df = pd.DataFrame(results)
                results_df.to_csv(p, index=True)
        
if __name__ == "__main__":
    main()