import anndata as ad
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_config import get_config
from monotonicity_e_distances_on_pertpy_data import get_e_distance_log


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
        p = os.path.join("/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_14_var_edistance", f"{p}_e_distance_results.csv")
        
        if os.path.exists(p):
            continue
        adata = ad.read_h5ad(f)
        group_by, reference = get_config(f)
        
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
                    results = get_e_distance_log(adata[list(subset_1) + list(subset_2)], group_by=group_by, reference=reference, subsampling=subsampling)   
                    results["group_by_split"] = group_by_split
                    results["group_by_split_reference"] = group_by_split_reference
                    results["test_group"] = test_group
                    all_results.append(results)
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(p, index=False)
        
if __name__ == "__main__":
    print("hello")
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji")