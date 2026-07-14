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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_config import get_config


def evaluate(name, data_path="/data/bionets/datasets/scrnaseq_ji/"):

    first_row = True

    subsets = [f for f in os.listdir(data_path) if (name in f) and f.endswith(".h5ad")]
            
    for f in subsets:
        # create a dedicated logger per file to ensure one logfile per dataset
        log_fname = f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_7_SSNR_effect_size/{os.path.basename(f).split('.')[0]}_results.txt"
        logger_name = os.path.basename(f).split('.')[0]
        logger = logging.getLogger(logger_name)
        # remove existing handlers if any (important when running in same process)
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        fh = logging.FileHandler(log_fname)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    
        adata = read_h5ad(os.path.join(data_path, f))

        group_by, reference = get_config(f)
            
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
                        logger.info(f"test_group,group_by,len_subset,{row}")
                        first_row = False
                        
                    row = ",".join([str(results[key]) for key in results])   
                    logger.info(f"{test_group},{group_by_split},{len(subset)},{row}")
                except:
                    logger.info(f"{test_group},{group_by},failed")
        # close handlers for this logger
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        first_row = True
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
        evaluate("processed_schiebinger", data_path=data_path)
    elif "mcfarland" in args.dataset:
        for dataset in ["processed_mcfarland_1.hdf5", "processed_mcfarland_2.hdf5", "processed_mcfarland_3.hdf5", "processed_mcfarland_4.hdf5", "processed_mcfarland_5.hdf5"]:
            evaluate(dataset, data_path=data_path)
    elif args.dataset == "norman":
        evaluate("processed_norman", data_path=data_path)
    elif args.dataset == "bhattacherjee":
        for dataset in ["processed_bhattacherjee_Astro.hdf5", "processed_bhattacherjee_Endo.hdf5", "processed_bhattacherjee_Excitatory.hdf5"]:
            evaluate(dataset, data_path=data_path)