import sys
sys.path.append("..")
from src.rosenbaum import *
import numpy as np
import scanpy as sc
import logging
import time
import pandas as pd
from anndata import read_h5ad
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



def scanpy_setup(adata):
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    else:
        adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.pca(adata, use_highly_variable=True)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def run_mcfarland(path="/data/bionets/datasets/scrnaseq_ji/mcfarland.hdf5"):
    nx = False
    if nx:
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
    
    for k in [5]:
        for i, test_group in enumerate([g6, g24]):
            start = time.time()
            p, z, s = rosenbaum(adata, group_by="perturbation", reference=["control"], test_group=test_group, rank=False, metric="sqeuclidean", k=k, nx=nx)    
            duration = time.time() - start
            logging.info(f"{np.unique(test_group)}; {k}; {p}; {z}; {s}; {duration:.6f}")


def run_norman(path="/data/bionets/datasets/scrnaseq_ji/norman.hdf5"):
    nx = False
    if nx:
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
    print(adata.obs["n_guides"].value_counts())

    logging.info(f"test_group; k; p; z; s; t")
    print(adata.obs["perturbation"].value_counts())
    
    for k in [2, 5]:
        for i, test_group in enumerate(["control", "1", "2"]):
            for j, reference in enumerate(["control", "1", "2"]):
                if j <= i:
                    continue
                start = time.time()
                p, z, s = rosenbaum(adata, group_by="n_guides", reference=reference, test_group=test_group, rank=False, metric="sqeuclidean", k=k, nx=nx)    
                duration = time.time() - start
                logging.info(f"{test_group}; {reference}; {k}; {p}; {z}; {s}; {duration:.6f}")


if __name__ == "__main__":
    run_mcfarland()