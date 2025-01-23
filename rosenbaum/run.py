import sys
sys.path.append("..")
from rosenbaum import *
import numpy as np
import scanpy as sc
import pandas as pd
from anndata import read_h5ad
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sc.set_figure_params(dpi=100, frameon=False, facecolor=None)

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


def run_mcfarland(path="/data_nfs/datasets/scrnaseq_ji/mcfarland.hdf5"):
    adata = read_h5ad(path)
    adata = scanpy_setup(adata)
    k = None
    p, z, s = rosenbaum(adata, group_by="perturbation", reference="control", test_group="Idasanutlin_6", rank=False, metric="sqeuclidean", k=k)    
    print("|", k, "|", p, "|", z, "|", s, "|")
    #adata = read_h5ad(path)
    #adata = scanpy_setup(adata)
    #print(rosenbaum(adata, group_by="perturbation", reference="control", test_group="Trametinib_24", rank=False, metric="sqeuclidean", k=5))


if __name__ == "__main__":
    run_mcfarland()