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
from importlib import reload


def evaluate(name, group_by, data_path="/data/bionets/datasets/scrnaseq_ji/"):
    logging.basicConfig(
        filename=f"../../../evaluation_results/1_7_monotonicity_SSNR_effect_size/{os.path.splitext(name)[0]}_results.txt",
        level=logging.INFO,
        format="%(message)s"
    )
    
    first_row = True

    subsets = [f for f in os.listdir(data_path) if (name in f) and f.endswith(".hdf5")]
            
    for f in subsets:
        adata = read_h5ad(os.path.join(data_path, f))

        if "mcfarland" in f:
            group_by = "pert_time"
            reference = "control"
            
        elif "norman" in f:
            group_by = "n_guides"
            reference = "control"
            
        elif "schiebinger" in f:
            group_by = "perturbation"
            reference = "control"
            
        elif "bhatta" in f:
            group_by = "label"
            reference = "Maintenance_Cocaine"
            
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        groups =  adata.obs[group_by].unique()

        for test_group in groups:
            if test_group == reference:
                continue    
            subset = adata[adata.obs[group_by].isin([test_group]), :].copy()
            for group_by_split in ["split_50", "split_10", "split_30"]:
                subset.obs[group_by_split] = subset.obs[group_by_split].astype(str)
                try:
                    results = test(subset, group_by=group_by_split, reference="0.0", test_group="1.0", rank=False, metric="sqeuclidean", k=100)    
                    if first_row:
                        row = ",".join(list(results.keys()))   
                        logging.info(f"test_group,group_by,len_subset,{row}")
                        first_row = False
                        
                    row = ",".join([str(results[key]) for key in results])   
                    logging.info(f"{test_group},{group_by_split},{len(subset)},{row}")
                except:
                    logging.info(f"{test_group},{group_by},failed")
    logging.shutdown()
    reload(logging)
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
        evaluate("processed_schiebinger", group_by="perturbation", data_path=data_path)
    elif "mcfarland" in args.dataset:
        for dataset in ["processed_mcfarland_1.hdf5", "processed_mcfarland_2.hdf5", "processed_mcfarland_3.hdf5", "processed_mcfarland_4.hdf5", "processed_mcfarland_5.hdf5"]:
            evaluate(dataset, group_by="pert_time", data_path=data_path)
    elif args.dataset == "norman":
        evaluate("processed_norman", group_by="n_guides", data_path=data_path)
    elif args.dataset == "bhattacherjee":
        for dataset in ["processed_bhattacherjee_Astro.hdf5", "processed_bhattacherjee_Endo.hdf5", "processed_bhattacherjee_Excitatory.hdf5"]:
            evaluate(dataset, group_by="label", data_path=data_path)