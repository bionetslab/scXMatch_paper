import sys
from scxmatch import test
import numpy as np
import os
import logging
from anndata import read_h5ad
import faulthandler
import argparse
faulthandler.enable()
np.random.seed(52)
from SSNR_on_pertpy_data import prepare


def evaluate(name, group_by, k, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/"):
    adata, group_by, reference, groups = prepare(data_path, name, "../evaluation_results/2_1_monotonicity_scxmatch/")  
    logging.info(f"test_group,group_by_split,group_by_split_reference,k,p,z,s")  
    for test_group in groups:
        if test_group == reference:
            continue
        for group_by_split in ["split_10", "split_30", "split_50"]:
            subset_1 = adata[( (adata.obs[group_by] == test_group) & (adata.obs[group_by_split] == 1) ), :].obs.index
            for group_by_split_reference in ["split_10", "split_30", "split_50"]:
                try:
                    subset_2 = adata[( (adata.obs[group_by] == reference) & (adata.obs[group_by_split_reference] == 1) ), :].obs.index
                    p, z, s = rosenbaum(adata[list(subset_1) + list(subset_2)], group_by=group_by, reference=reference, test_group=test_group, rank=rank, metric=metric, k=k)    
                    logging.info(f"{test_group},{group_by_split},{group_by_split_reference},{k},{p},{z},{s}")    
                except:
                    logging.info(f"{test_group},{group_by_split},{group_by_split_reference},failed")
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser("run")
    parser.add_argument("dataset", type=str, choices=["schiebinger", "mcfarland", "norman", "sciplex_A549", "sciplex_K562", "sciplex_MCF7", 'bhattacherjee',
                                                      "LSI_Liscovitch-BrauerSanjana2021_K562_2.hdf5", "LSI_PierceGreenleaf2021_K562.hdf5", 
                                                      "LSI_MimitouSmibert2021.hdf5", "LSI_PierceGreenleaf2021_MCF7.hdf5", "LSI_PierceGreenleaf2021_GM12878.hdf5", 
                                                      "LSI_Liscovitch-BrauerSanjana2021_K562_1.hdf5", "processed_PierceGreenleaf2021_GM12878.hdf5", 
                                                      "processed_Liscovitch-BrauerSanjana2021_K562_1.hdf5", "processed_PierceGreenleaf2021_K562.hdf5", 
                                                      "processed_Liscovitch-BrauerSanjana2021_K562_2.hdf5", "processed_MimitouSmibert2021.hdf5", "processed_PierceGreenleaf2021_MCF7.hdf5"])
    args = parser.parse_args()
    
    dataset = args.dataset
    metric = "sqeuclidean"
    k = 100
    data_path= "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/"

    if args.dataset == "schiebinger":
        evaluate("processed_schiebinger", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "mcfarland":
        evaluate("processed_mcfarland", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "norman":
        evaluate("processed_norman", group_by="n_guides", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "sciplex_A549":
        evaluate("processed_sciplex_A549", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "sciplex_K562":
        evaluate("processed_sciplex_K562", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "sciplex_MCF7":
        evaluate("processed_sciplex_MCF7", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "bhattacherjee":
        print("BHATTACHERJEE")
        evaluate("processed_bhattacherjee_excitatory", group_by="label", rank=False, metric=metric, data_path=data_path, k=k)
    elif "LSI" in args.dataset:
        evaluate(args.dataset, group_by="perturbation", rank=False, metric="euclidean", data_path=data_path, k=k)
    else:
        evaluate(args.dataset, group_by="perturbation", rank=False, metric="sqeuclidean", data_path=data_path, k=k)