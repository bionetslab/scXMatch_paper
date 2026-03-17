import sys
from scxmatch import test
import numpy as np
import scanpy as sc
import os
import logging
import time
from anndata import read_h5ad
import argparse
from SSNR_on_pertpy_data import prepare

def evaluate(name, group_by, k, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/"):
    adata, group_by, reference, groups = prepare(data_path, name, result_path="/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_1_monotonicity_scxmatch/")   
    adata.obs["perturbation2"] = adata.obs["perturbation"].apply(lambda x: x.split("_")[0])
    
    print(adata.obs["perturbation2"].unique())
    for compound in adata.obs["perturbation2"].unique():
        
        if compound != "control":
            subset = adata[adata.obs["perturbation2"].isin([compound, "control"])].copy()
        print(subset.obs[group_by].unique())
        if len(subset.obs[group_by].unique()) > 4:
            for test_group in groups:
                if test_group == reference:
                    continue
                p, z, s = test(subset, group_by=group_by, reference=str(reference), test_group=str(test_group), rank=rank, metric=metric, k=k)    
                logging.info(f"{compound},{test_group},{reference},{k},{p},{z},{s}")
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser("run")
    parser.add_argument("dataset", type=str, choices=["schiebinger", "mcfarland", "norman", "sciplex_A549", "sciplex_K562", "sciplex_MCF7", "bhattacherjee",
                                                      "LSI_Liscovitch-BrauerSanjana2021_K562_2.hdf5", "LSI_PierceGreenleaf2021_K562.hdf5", 
                                                      "LSI_MimitouSmibert2021.hdf5", "LSI_PierceGreenleaf2021_MCF7.hdf5", "LSI_PierceGreenleaf2021_GM12878.hdf5", 
                                                      "LSI_Liscovitch-BrauerSanjana2021_K562_1.hdf5", "processed_PierceGreenleaf2021_GM12878.hdf5", 
                                                      "processed_Liscovitch-BrauerSanjana2021_K562_1.hdf5", "processed_PierceGreenleaf2021_K562.hdf5", 
                                                      "processed_Liscovitch-BrauerSanjana2021_K562_2.hdf5", "processed_MimitouSmibert2021.hdf5", "processed_PierceGreenleaf2021_MCF7.hdf5"])
    args = parser.parse_args()
    k = 100
    metric = "euclidean"
    data_path= "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/"

    if args.dataset == "schiebinger":
        evaluate("processed_schiebinger", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "mcfarland":
        evaluate("processed_mcfarland", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "bhattacherjee":
        print("BHATTACHERJEE")
        evaluate("processed_bhattacherjee_excitatory", group_by="label", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "norman":
        evaluate("processed_norman", group_by="n_guides", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "sciplex_A549":
        evaluate("processed_sciplex_A549", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "sciplex_K562":
        evaluate("processed_sciplex_K562", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "sciplex_MCF7":
        evaluate("processed_sciplex_MCF7", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=k)
    elif "LSI" in args.dataset:
        evaluate(args.dataset, group_by="perturbation", rank=False, metric="euclidean", data_path=data_path, k=k)
    else:
        evaluate(args.dataset, group_by="perturbation", rank=False, metric="sqeuclidean", data_path=data_path, k=k)