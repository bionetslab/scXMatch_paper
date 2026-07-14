from scxmatch import test
import numpy as np
import os
import logging
from anndata import read_h5ad
import faulthandler
import argparse
faulthandler.enable()
np.random.seed(52)
from importlib import reload


def evaluate(name, group_by, k, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/"):
    subsets = [f for f in os.listdir(data_path) if (name in f) and f.endswith(".h5ad")]
    for f in subsets:
        if "mcfarland" in name:
            group_by = "pert_time"
            reference = "control"
            
        elif "norman_" in name:
            group_by = "label"
            reference = 0
            
        elif "schiebinger" in name:
            group_by = "perturbation"
            reference = "control"
            
        elif "bhatta" in name:
            group_by = "label"
            reference = "Maintenance_Cocaine"
        
        log_file = f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_8_SV_effect_size/{os.path.basename(f).split('.')[0]}_results.txt"
        logger = logging.getLogger(f"xmatch_{os.path.basename(f)}")
        logger.propagate = False
        logger.setLevel(logging.INFO)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        adata = read_h5ad(os.path.join(data_path, f))
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        groups =  adata.obs[group_by].unique()
        for test_group in groups:
            if test_group == reference:
                continue
            for group_by_split in ["split_10", "split_30", "split_50"]:
                subset_1 = adata[( (adata.obs[group_by] == test_group) & (adata.obs[group_by_split] == 1) ), :].obs.index
                for group_by_split_reference in ["split_10", "split_30", "split_50"]:
                    try:
                        subset_2 = adata[( (adata.obs[group_by] == reference) & (adata.obs[group_by_split_reference] == 1) ), :].obs.index
                        results = test(adata[list(subset_1) + list(subset_2)], group_by=group_by, reference=reference, test_group=test_group, rank=rank, metric=metric, k=k)   
                        row = ",".join([str(results[key]) for key in results])    
                        logger.info(f"{test_group},{group_by_split},{group_by_split_reference},{k},{row}")    
                    except:
                        logger.info(f"{test_group},{group_by_split},{group_by_split_reference},failed")
        handler.close()
        logger.removeHandler(handler)
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser("run")
    parser.add_argument("dataset", type=str)

    args = parser.parse_args()
    
    dataset = args.dataset
    metric = "sqeuclidean"
    k = 100
    data_path= "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/"

    if args.dataset == "schiebinger":
        evaluate("processed_schiebinger", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=k)
    elif "mcfarland" in args.dataset:
        for dataset in ["processed_mcfarland_1.hdf5", "processed_mcfarland_2.hdf5", "processed_mcfarland_3.hdf5", "processed_mcfarland_4.hdf5", "processed_mcfarland_5.hdf5"]:
            evaluate(dataset, group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "norman":
        evaluate("processed_norman", group_by="n_guides", rank=False, metric=metric, data_path=data_path, k=k)
    elif args.dataset == "bhattacherjee":
        for dataset in ["processed_bhattacherjee_Astro.hdf5", "processed_bhattacherjee_Endo.hdf5", "processed_bhattacherjee_Excitatory.hdf5"]:
            evaluate(dataset, group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=k)
