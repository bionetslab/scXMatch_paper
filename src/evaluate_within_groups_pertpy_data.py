import sys
sys.path.append("..")
sys.path.append("../../scxmatch/src")
from scxmatch import rosenbaum

import numpy as np
import scanpy as sc
import os
import logging
import time
from anndata import read_h5ad
import faulthandler
import argparse
faulthandler.enable()
np.random.seed(52)

def scanpy_setup(adata):
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    else:
        adata.layers['counts'] = adata.X.copy()
    #sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def evaluate(name, group_by, k, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/"):
    log_name = f"../evaluation_results/within_{name}_results.txt"    
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="%(message)s"
    )
    logging.info(f"test_group,reference,k,p,z,s,d,N")

    file_name = os.path.join(data_path, f"{name}.hdf5")
    print(file_name)
    adata = read_h5ad(file_name)
    adata = scanpy_setup(adata)
    
    if name == "mcfarland":
        g6 = list(np.unique([v for v in adata.obs["perturbation"].values if v.endswith("_6")]))
        g24 = list(np.unique([v for v in adata.obs["perturbation"].values if v.endswith("_24")]))
        print(adata.obs[group_by].value_counts())
        test_groups = [["control"]]
        reference = "control"
        
    elif name == "norman":
        adata.obs['n_guides'] = np.where(
            adata.obs["perturbation"].str.contains("control"),
            "control",  # If true, assign "control"
            adata.obs["perturbation"].str.count("\+") + 1)    
        test_groups = [["control"]]
        reference = "control"
        
    elif "sciplex" in name:
        test_groups = [[1000.0]]
        reference = 0.0
   
    elif "schiebinger" in name:
        test_groups = ["control", 'D1.5', 'D2', 'D2.5', 'D3', 'D3.5', 'D4', 'D4.5', 'D5', 'D5.5', 'D6', 'D6.5', 'D7', 'D7.5', 'D8', 'D8.5', 'D9', 'D9.5', 'D10', 'D10.5', 'D11', 'D11.5', 'D12', 'D12.5', 'D13', 'D13.5', 'D14', 'D14.5', 'D15', 'D15.5', 'D16', 'D16.5', 'D17', 'D17.5', 'D18']
        reference = "control"
    else:
        raise ValueError("Unknown dataset")

    for test_group in test_groups:
        if "schiebinger" in name:
            test_group = [test_group]

        subset = adata[adata.obs[group_by].isin(test_group), :].copy()
        print(subset)
        for prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
            while True:
                try:
                    subset.obs["scxmatch_testing"] = np.random.choice(["0", "1"], size=len(subset), p=[prob, 1-prob])
                    n, m = subset.obs["scxmatch_testing"].value_counts().values
                    break
                except:
                    continue
            p, z, s = rosenbaum(subset, group_by="scxmatch_testing", reference="0", test_group="1", rank=rank, metric=metric, k=k)    
            logging.info(f"{test_group},{test_group},{k},{p},{z},{s},{prob},{len(subset)}")
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser("run")
    parser.add_argument("dataset", type=str, choices=["schiebinger", "mcfarland", "norman", "sciplex_A549", "sciplex_K562", "sciplex_MCF7"])
    args = parser.parse_args()
    
    dataset = args.dataset
    metric = "sqeuclidean"
    data_path= "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/"

    if args.dataset == "schiebinger":
        evaluate("schiebinger", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "mcfarland":
        evaluate("mcfarland", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "norman":
        evaluate("norman", group_by="n_guides", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "sciplex_A549":
        evaluate("sciplex_A549", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "sciplex_K562":
        evaluate("sciplex_K562", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "sciplex_MCF7":
        evaluate("sciplex_MCF7", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=100)
