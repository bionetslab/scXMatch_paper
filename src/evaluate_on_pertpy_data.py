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


def run_mcfarland(path="/data/bionets/datasets/scrnaseq_ji/mcfarland.hdf5", use_nx=False):
    if use_nx:
        logging.basicConfig(
            filename="../evaluation_results/mcfarland_nx_log.txt",
            level=logging.INFO,
            format="%(message)s"
        )
    else:
        logging.basicConfig(
            filename="../evaluation_results/mcfarland_gt_log.txt",
            level=logging.INFO,
            format="%(message)s"
        )
    
    adata = read_h5ad(path)
    adata = scanpy_setup(adata)

    g6 = [v for v in adata.obs["perturbation"].values if v.endswith("_6")]
    g24 = [v for v in adata.obs["perturbation"].values if v.endswith("_24")]

    logging.info(f"test_group; k; p; z; s; t")
    print(adata.obs["perturbation"].value_counts())
    
    for k in [2, 5, 10]:
        for i, test_group in enumerate([g6, g24]):
            start = time.time()
            p, z, s = rosenbaum(adata, group_by="perturbation", reference=["control"], test_group=test_group, rank=False, metric="sqeuclidean", k=k, use_nx=use_nx)    
            duration = time.time() - start
            logging.info(f"{np.unique(test_group)}; {k}; {p}; {z}; {s}; {duration:.6f}")


def run_norman(path="/data/bionets/datasets/scrnaseq_ji/norman.hdf5", use_nx=False):
    if use_nx:
        logging.basicConfig(
            filename="../evaluation_results/norman_nx_log.txt",
            level=logging.INFO,
            format="%(message)s"
        )
    else:
        logging.basicConfig(
            filename="../evaluation_results/norman_gt_log.txt",
            level=logging.INFO,
            format="%(message)s"
        )
    
    adata = read_h5ad(path)
    adata = scanpy_setup(adata)

    print(adata.obs["n_guides"].value_counts())
    adata.obs['n_guides'] = np.where(
        adata.obs["perturbation"].str.contains("control"),  # Check if "control" is in the perturbation
        "control",  # If true, assign "control"
        adata.obs["perturbation"].str.count("\+") + 1  # Otherwise, count "+" and add 1
    )    

    logging.info(f"test_group; k; p; z; s; t")
        
    reference = "control"
    reference_subset = sc.pp.subsample(adata[adata.obs["n_guides"] == reference], n_obs=1000, copy=True)

    for k in [2, 5, 10]:
        for i, test_group in enumerate(["1", "2"]):
            test_subset = sc.pp.subsample(adata[adata.obs["n_guides"] == test_group], n_obs=1000, copy=True)
            subset = ad.concat([reference_subset, test_subset])
            start = time.time()
            p, z, s = rosenbaum(subset.copy(), group_by="n_guides", reference=reference, test_group=test_group, rank=False, metric="sqeuclidean", k=k, use_nx=use_nx)    
            duration = time.time() - start
            logging.info(f"{test_group}; {reference}; {k}; {p}; {z}; {s}; {duration:.6f}")



def run_sciplex(path="/data/bionets/datasets/scrnaseq_ji/sciplex_A549.hdf5", use_nx=False):
    name = os.path.splitext(os.path.basename(path))[0]

    if use_nx:
        logging.basicConfig(
            filename=f"../evaluation_results/{name}_nx_log.txt",
            level=logging.INFO,
            format="%(message)s"
        )
    else:
        logging.basicConfig(
            filename=f"../evaluation_results/{name}_gt_log.txt",
            level=logging.INFO,
            format="%(message)s"
    )
    
    adata = read_h5ad(path)
    adata = scanpy_setup(adata)
    reference = 0.0

    reference_subset = sc.pp.subsample(adata[adata.obs["dose_value"] == 0.0], n_obs=1000, copy=True)

    for k in [2, 5, 10]:
        for test_group in [10.0, 100.0, 1000.0, 10000.0]:
            test_subset = sc.pp.subsample(adata[adata.obs["dose_value"] == test_group], n_obs=1000, copy=True)
            subset = ad.concat([reference_subset, test_subset])
            print(subset)
            start = time.time()
            p, z, s = rosenbaum(subset.copy(), group_by="dose_value", reference=[reference], test_group=[test_group], rank=False, metric="sqeuclidean", k=k, use_nx=use_nx)    
            duration = time.time() - start
            logging.info(f"{test_group}; {reference}; {k}; {p}; {z}; {s}; {duration:.6f}")



def run_schiebinger(path="/data/bionets/datasets/scrnaseq_ji/schiebinger.hdf5", use_nx=False):
    if use_nx:
        logging.basicConfig(
            filename="../evaluation_results/schiebinger_nx_log.txt",
            level=logging.INFO,
            format="%(message)s"
        )
    else:
        logging.basicConfig(
            filename="../evaluation_results/schiebinger_gt_log.txt",
            level=logging.INFO,
            format="%(message)s"
        )
    
    adata = read_h5ad(path)
    adata = scanpy_setup(adata)

    logging.info(f"test_group; k; p; z; s; t")
    print(sorted(np.unique(adata.obs["perturbation"].values)))
       
    reference = "control"
    reference_subset = sc.pp.subsample(adata[adata.obs["perturbation"] == "control"], n_obs=1000, copy=True)

    for k in [5, 10]:
        for test_group in ['D1.5', 'D2', 'D2.5', 'D3', 'D3.5', 'D4', 'D4.5', 'D5', 'D5.5', 'D6', 'D6.5', 'D7', 'D7.5', 'D8', 'D8.5', 'D9', 'D9.5', 'D10', 'D10.5', 'D11', 'D11.5', 'D12', 'D12.5', 'D13', 'D13.5', 'D14', 'D14.5', 'D15', 'D15.5', 'D16', 'D16.5', 'D17', 'D17.5', 'D18']:
            print("test group")
            test_subset = sc.pp.subsample(adata[adata.obs["perturbation"] == test_group], n_obs=1000, copy=True)
            subset = ad.concat([reference_subset, test_subset])
            start = time.time()
            p, z, s = rosenbaum(subset.copy(), group_by="perturbation", reference=reference, test_group=test_group, rank=False, metric="sqeuclidean", k=k, use_nx=use_nx)    
            duration = time.time() - start
            logging.info(f"total; {test_group}; {reference}; {k}; {p}; {z}; {s}; {duration:.6f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("run")
    parser.add_argument("dataset", type=str, choices=["schiebinger", "mcfarland", "norman", "sciplex_A549", "sciplex_K562", "sciplex_MCF7"])
    parser.add_argument("use_nx", type=str, choices=["False", "True"])
    args = parser.parse_args()
    
    use_nx = True if args.use_nx == "True" else False
    dataset = args.dataset

    print(dataset, use_nx)

    if args.dataset == "schiebinger":
         run_schiebinger(path="/data/bionets/datasets/scrnaseq_ji/schiebinger.hdf5", use_nx=use_nx)
    elif args.dataset == "mcfarland":
        run_mcfarland(path="/data/bionets/datasets/scrnaseq_ji/mcfarland.hdf5", use_nx=use_nx)
    elif args.dataset == "norman":
        run_norman(path="/data/bionets/datasets/scrnaseq_ji/norman.hdf5", use_nx=use_nx)
    elif args.dataset == "sciplex_A549":
        run_sciplex(path="/data/bionets/datasets/scrnaseq_ji/sciplex_A549.hdf5", use_nx=use_nx)
    elif args.dataset == "sciplex_K562":
        run_sciplex(path="/data/bionets/datasets/scrnaseq_ji/sciplex_K562.hdf5", use_nx=use_nx)
    elif args.dataset == "sciplex_MCF7":
        run_sciplex(path="/data/bionets/datasets/scrnaseq_ji/sciplex_MCF7.hdf5", use_nx=use_nx)
