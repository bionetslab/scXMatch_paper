import sys
sys.path.append("..")
from src.rosenbaum import *
import numpy as np
import scanpy as sc
import os
import logging
import time
from anndata import read_h5ad
import faulthandler
import argparse
faulthandler.enable()


def scanpy_setup(adata):
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    else:
        adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def evaluate(name, use_nx, group_by, ks=[5], sub_sample_size=1000, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/"):
    method = "nx" if use_nx else "gt"
    log_name = f"../evaluation_results/{name}_{method}_subsampled_{sub_sample_size}_log.txt"    
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="%(message)s"
    )
    
    file_name = os.path.join(data_path, f"{name}.hdf5")
    adata = read_h5ad(file_name)
    adata = scanpy_setup(adata)

    if name == "mcfarland":
        g6 = list(np.unique([v for v in adata.obs["perturbation"].values if v.endswith("_6")]))
        g24 = list(np.unique([v for v in adata.obs["perturbation"].values if v.endswith("_24")]))
        test_groups = [g6, g24]
        reference = "control"
        
    elif name == "norman":
        adata.obs['n_guides'] = np.where(
            adata.obs["perturbation"].str.contains("control"),
            "control",  # If true, assign "control"
            adata.obs["perturbation"].str.count("\+") + 1)    
        test_groups = [["1"], ["2"]]
        reference = "control"
        
    elif "sciplex" in name:
        test_groups = [[10.0], [100.0], [1000.0], [10000.0]]
        reference = 0.0
   
    elif "schiebinger" in name:
        test_groups = ['D1.5', 'D2', 'D2.5', 'D3', 'D3.5', 'D4', 'D4.5', 'D5', 'D5.5', 'D6', 'D6.5', 'D7', 'D7.5', 'D8', 'D8.5', 'D9', 'D9.5', 'D10', 'D10.5', 'D11', 'D11.5', 'D12', 'D12.5', 'D13', 'D13.5', 'D14', 'D14.5', 'D15', 'D15.5', 'D16', 'D16.5', 'D17', 'D17.5', 'D18']
        reference = "control"
    else:
        raise ValueError("Unknown dataset")

        
    if sub_sample_size:
        if len(adata[adata.obs[group_by] == reference]) > sub_sample_size:
            reference_subset = sc.pp.subsample(adata[adata.obs[group_by] == reference], n_obs=sub_sample_size, copy=True)
        else:
            reference_subset = adata[adata.obs[group_by] == reference]

    results = list()
    for k in ks:
        for i, test_group in enumerate(test_groups):
            if sub_sample_size:
                if not isinstance(test_group, list):
                    test_group = [test_group]
                if len(adata[adata.obs[group_by].isin(test_group)]) > sub_sample_size:
                    test_subset = sc.pp.subsample(adata[adata.obs[group_by].isin(test_group)], n_obs=sub_sample_size, copy=True)
                else:
                    test_subset = adata[adata.obs[group_by].isin(test_group)]
                subset = ad.concat([reference_subset, test_subset])
            else:
                subset = adata
            start = time.time()
            p, z, s = rosenbaum(subset.copy(), group_by=group_by, reference=reference, test_group=test_group, rank=rank, metric=metric, k=k, use_nx=use_nx)    
            duration = time.time() - start
            logging.info(f"{test_group}; {reference}; {k}; {p}; {z}; {s}; {duration:.6f}")
            results.append([test_group, reference, k, p, z, s, duration])
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser("run")
    parser.add_argument("dataset", type=str, choices=["schiebinger", "mcfarland", "norman", "sciplex_A549", "sciplex_K562", "sciplex_MCF7"])
    parser.add_argument("use_nx", type=str, choices=["False", "True"])
    parser.add_argument("sub_sample_sizes", type=int)

    args = parser.parse_args()
    
    use_nx = True if args.use_nx == "True" else False
    dataset = args.dataset
    sub_sample_size = args.sub_sample_sizes
    print(dataset, use_nx)

    if args.dataset == "schiebinger":
        evaluate("schiebinger", use_nx=True, group_by="perturbation", sub_sample_size=sub_sample_size, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/")
    elif args.dataset == "mcfarland":
        evaluate("mcfarland", use_nx=True, group_by="perturbation", sub_sample_size=sub_sample_size, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/")
    elif args.dataset == "norman":
        evaluate("norman", use_nx=True, group_by="n_guides", sub_sample_size=sub_sample_size, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/")
    elif args.dataset == "sciplex_A549":
        evaluate("sciplex_A549", use_nx=True, group_by="dose_value", sub_sample_size=sub_sample_size, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/")
    elif args.dataset == "sciplex_K562":
        evaluate("sciplex_K562", use_nx=True, group_by="dose_value", sub_sample_size=sub_sample_size, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/")
    elif args.dataset == "sciplex_MCF7":
        evaluate("sciplex_MCF7", use_nx=True, group_by="dose_value", sub_sample_size=sub_sample_size, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/")
