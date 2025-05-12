import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm

def scanpy_setup(adata):
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    else:
        adata.layers['counts'] = adata.X.copy()
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def assign_pseudo_bulks(adata, group_by, bulk_size, label='pseudo_bulk'):
    """
    Randomly assigns all cells in `adata` to pseudo-bulks within each `group_by` condition.
    Each pseudo-bulk contains up to `bulk_size` cells. The last one may be smaller.
    
    Parameters:
        adata: AnnData object
        group_by: str, column in adata.obs to group by (e.g., 'perturbation')
        bulk_size: int, number of cells per pseudo-bulk
        label: str, name of the new column in adata.obs to store bulk assignment
        
    Returns:
        Modified AnnData with a new `.obs` column `label`
    """
    np.random.seed(42)  # for reproducibility
    bulk_labels = pd.Series(index=adata.obs_names, dtype="object")

    for condition, idx in adata.obs.groupby(group_by).groups.items():
        cell_indices = np.array(idx)
        np.random.shuffle(cell_indices)

        n_bulks = int(np.ceil(len(cell_indices) / bulk_size))

        for i in range(n_bulks):
            start = i * bulk_size
            end = min((i + 1) * bulk_size, len(cell_indices))
            bulk_cells = cell_indices[start:end]
            bulk_label = f"{condition}_bulk{i+1}"
            bulk_labels.loc[bulk_cells] = bulk_label

    adata.obs[label] = bulk_labels
    return adata


def assign_random_split_within_group(adata, group_by, p=0.5, label="split"):
    """
    Randomly assigns a binary label (0 or 1) to cells within each group in `group_by`.
    Each cell has probability `p` of being assigned 1, and (1 - p) of being 0.
    
    Parameters:
        adata: AnnData object
        group_by: str, column in adata.obs to group by (e.g., 'perturbation')
        p: float, probability of assigning 1
        label: str, name of the new column in adata.obs to store the label

    Returns:
        Modified AnnData with a new `.obs` column `label` containing 0 or 1
    """
    np.random.seed(42)  # For reproducibility
    labels = pd.Series(index=adata.obs_names, dtype="int")

    for condition, idx in adata.obs.groupby(group_by).groups.items():
        cell_indices = np.array(idx)
        assignments = np.random.choice([0, 1], size=len(cell_indices), p=[1-p, p])
        labels.loc[cell_indices] = assignments

    adata.obs[label] = labels.astype("category")
    return adata



def assign_pseudo_bulks_and_splits(adata, group_by):
    adata = assign_pseudo_bulks(adata, group_by=group_by, bulk_size=100, label='pseudo_bulk_100')
    adata = assign_pseudo_bulks(adata, group_by=group_by, bulk_size=200, label='pseudo_bulk_200')
    adata = assign_pseudo_bulks(adata, group_by=group_by, bulk_size=500, label='pseudo_bulk_500')
    adata = assign_random_split_within_group(adata, group_by=group_by, p=0.5, label="split_50")
    adata = assign_random_split_within_group(adata, group_by=group_by, p=0.4, label="split_40")
    adata = assign_random_split_within_group(adata, group_by=group_by, p=0.3, label="split_30")
    adata = assign_random_split_within_group(adata, group_by=group_by, p=0.2, label="split_20")
    adata = assign_random_split_within_group(adata, group_by=group_by, p=0.1, label="split_10")
    adata = assign_random_split_within_group(adata, group_by=group_by, p=0.01, label="split_1")
    return adata


def pre_process_all(dataset_path):
    names = [f for f in os.listdir(dataset_path) if (f.endswith("hdf5") and "bhatta" in f)]
    files = [os.path.join(dataset_path, f) for f in names]
    
    print(names)
    for n, f in tqdm(zip(names, files)):
        adata = ad.read_h5ad(f)
        adata = scanpy_setup(adata)
        
        if "mcfarland" in n:
            group_by = "perturbation_grouped"
            adata.obs[group_by] = adata.obs['perturbation'].apply(lambda x: x.split("_")[-1])
        elif "norman" in n:
            group_by = "n_guides"
            adata.obs['n_guides'] = np.where(
            adata.obs["perturbation"].str.contains("control"),
                                "control",  # If true, assign "control"
                                adata.obs["perturbation"].str.count("\+") + 1)    
        elif "sciplex" in n:
            group_by = "dose_value"
        elif "schiebinger" in n:
            group_by = "perturbation"
        elif "bhattacherjee" in n:
            group_by = "label"
        else:
            raise ValueError("Unknown dataset")
        
        adata = assign_pseudo_bulks_and_splits(adata, group_by)
        adata.write_h5ad(f"{dataset_path}/processed_{n}")
    
    
if __name__ == "__main__":
    pre_process_all("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/raw")