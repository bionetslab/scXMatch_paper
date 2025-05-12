import sys
sys.path.append("..")
sys.path.append("../../scxmatch/src")
from scxmatch.rosenbaum import rosenbaum
import numpy as np
import os
import logging
from anndata import read_h5ad
import faulthandler
import argparse
faulthandler.enable()
np.random.seed(52)


def prepare(data_path, name, result_path=f"../evaluation_results/1_2_SSNR_scxmatch/"):
    log_name = os.path.join(result_path, f"{name}_results.txt")
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="%(message)s"
    )

    if "hdf5" not in name:
        file_name = os.path.join(data_path, f"{name}.hdf5")
    else:
        file_name = os.path.join(data_path, name)

    adata = read_h5ad(file_name)

    if "mcfarland" in name:
        group_by = "perturbation_grouped"
        reference = "control"
        
    elif "norman" in name:
        group_by = "n_guides"
        reference = "control"
        
    elif "sciplex" in name:
        group_by = "dose_value"
        reference = "0.0"
        
    elif "schiebinger" in name:
        group_by = "perturbation"
        reference = "control"
    
    elif "bhattacherjee" in name:
        group_by = "label"
        reference = "Maintenance_Cocaine"
        
    else:
        group_by = "perturbation"
        reference = "control"

    adata.obs[group_by] = adata.obs[group_by].astype(str)
    groups = adata.obs[group_by].unique()
    return adata, group_by, reference, groups


def evaluate(name, group_by, k, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/"):
    adata, group_by, reference, groups = prepare(data_path, name)  
    
    for test_group in groups:
        subset = adata[adata.obs[group_by].isin([test_group]), :].copy()
        for group_by_split in ["split_50", "split_10", "split_30"]:
            subset.obs[group_by_split] = subset.obs[group_by_split].astype(str)
            try:
                p, z, s = rosenbaum(subset, group_by=group_by_split, reference="0.0", test_group="1.0", rank=rank, metric=metric, k=k)    
                logging.info(f"{test_group},{group_by_split},{k},{p},{z},{s},{len(subset)}")
            except:
                logging.info(f"{test_group},{group_by},failed")
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