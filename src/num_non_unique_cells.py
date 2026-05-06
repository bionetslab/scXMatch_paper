# Code underlying the claims made in the first rebuttal response to comment 2.3 
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def num_non_unique_cells(adata: ad.AnnData) -> int:
    X = adata.X

    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    # Create a hashable representation of each row
    row_hashes = np.array([
        hash((tuple(X[i].indices), tuple(X[i].data)))
        for i in range(X.shape[0])
    ])

    # Count duplicates
    _, counts = np.unique(row_hashes, return_counts=True)
    total_duplicate_cells = np.sum(counts[counts > 1])
    return total_duplicate_cells


def main():
    datasets = [
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_bhattacherjee_Astro.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_bhattacherjee_Endo.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_bhattacherjee_Excitatory.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_mcfarland_1.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_mcfarland_2.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_mcfarland_3.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_mcfarland_4.hdf5",    
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_mcfarland_5.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_norman.hdf5",
        "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/processed_schiebinger.hdf5"
    ]
    
    for dataset in datasets:
        adata = sc.read_h5ad(dataset)
        num_duplicates = num_non_unique_cells(adata)
        print(f"Number of duplicate cells: {num_duplicates} / {len(adata)} in {dataset}")
    
    test = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]]) # -> 4/5
    test_adata = ad.AnnData(X=csr_matrix(test))
    num_duplicates_test = num_non_unique_cells(test_adata)
    print(f"Number of duplicate cells in test: {num_duplicates_test} / {len(test_adata)}")
    
    
if __name__ == "__main__":
    main()
    
    