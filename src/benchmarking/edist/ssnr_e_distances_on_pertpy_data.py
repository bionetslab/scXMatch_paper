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
        p = os.path.join("/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_13_SSNR_edistance", f"{p}_e_distance_results.csv")
        
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
            subset = adata[adata.obs[group_by].isin([test_group]), :].copy()
            for group_by_split in ["split_50", "split_10", "split_30"]:
                subset.obs[group_by_split] = subset.obs[group_by_split].astype(str)
                log = get_e_distance_log(subset, group_by=group_by_split, reference="0.0", subsampling=subsampling)
                log["test_group"] = test_group
                log["group_by_split"] = group_by_split
                log["len_subset"] = len(subset)
                all_results.append(log)
                    
        pd.concat(all_results).to_csv(p, index=False)

        
if __name__ == "__main__":
    print("hello")
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji")