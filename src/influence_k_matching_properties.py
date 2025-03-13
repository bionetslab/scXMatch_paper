from anndata import read_h5ad
import numpy as np
import os
import pandas as pd 
import scanpy as sc
import logging

import sys
sys.path.append("../../scxmatch/")
from src.scxmatch import *
sys.modules.keys()

np.random.seed(42)
sc._settings.ScanpyConfig.n_jobs = -1

def subsample_adata(adata, sample_size):
    sampled_indices = np.random.choice(range(len(adata)), sample_size, replace=False)    
    adata_subsampled = adata[sampled_indices, :].copy()
    return adata_subsampled


def compute_matching_cost(G, matching):
    """ Computes the total weight of the given matching. """
    cost = 0
    for u,v in matching:
        cost += G.edge_properties['weight'][u, v]
    return cost / len(matching)


def run_tests(adata, group_by, reference, test_group, ks, subsample_sizes):
    for subsample_size in subsample_sizes:
        if subsample_size:
            subsampled = subsample_adata(adata, subsample_size)
        else:
            subsampled = adata
        (ref_p, ref_z, ref_s), ref_G, ref_matching = rosenbaum.rosenbaum(subsampled, group_by=group_by, reference=reference, test_group=test_group, k=None, return_matching=True)

        ref_cost = compute_matching_cost(ref_G, ref_matching)
        logging.info(f"None, {reference}, {test_group}, {ref_p}, {ref_z}, {ref_s}, {ref_cost}, {len(ref_G.get_edges())}, {subsample_size}")

        for k in ks:
            print(k)
            if len(subsampled) - k > 500:
                (p, z, s), G, matching = rosenbaum.rosenbaum(subsampled, group_by=group_by, reference=reference, test_group=test_group, k=k, return_matching=True)
                edge_cost = compute_matching_cost(G, matching)
                logging.info(f"{k}, {reference}, {test_group}, {p}, {z}, {s}, {edge_cost}, {len(G.get_edges())}, {subsample_size}")

        

    
def main(data_path = "/data_nfs/datasets/scrnaseq_ji"):
    adata = read_h5ad(os.path.join(data_path, "sciplex_MCF7.hdf5"))
    group_by = "dose_value"
    reference = 0.0

    adata = utils.scanpy_setup(adata)

    logging.basicConfig(
        filename=f"../evaluation_results/sciplex_MCF7_matching_properties.txt",
        level=logging.INFO,
        format="%(message)s"
    )

    logging.info(f"k, ref, test, p, z, s, cost, #edges, #nodes")

    adata.obs[group_by] = adata.obs[group_by].astype('category')
    groups = list(adata.obs[group_by].unique())

    for group in groups:
        if group != reference:
            subset = adata[adata.obs[group_by].isin([reference, group])]
            run_tests(adata=subset, 
                      group_by=group_by, 
                      reference=reference, 
                      test_group=group, 
                      ks=[5, 10, 20, 50, 100, 500, 1000], 
                      subsample_sizes=[100, 500, 1000, 2500, 5000, 10000, 20000, 50000, None])

                
if __name__ == "__main__":
    main()
