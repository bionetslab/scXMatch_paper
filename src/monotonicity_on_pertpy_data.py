import sys
sys.path.append("..")
sys.path.append("../../scxmatch/src")
from scxmatch import rosenbaum

import numpy as np
import scanpy as sc
import os
import logging
import time
from anndata import read_h5ad
import faulthandler
import argparse
faulthandler.enable()
from SSNR_on_pertpy_data import prepare

def evaluate(name, group_by, k, rank=False, metric="sqeuclidean", data_path="/data/bionets/datasets/scrnaseq_ji/"):
    adata, group_by, reference, groups = prepare(data_path, name, result_path="../evaluation_results/1_1_monotonicity_scxmatch/")        
    for i, test_group in enumerate(groups):
        if test_group == reference:
            continue
        p, z, s = rosenbaum(adata.copy(), group_by=group_by, reference=reference, test_group=test_group, rank=rank, metric=metric, k=k)    
        logging.info(f"{test_group},{reference},{k},{p},{z},{s}")
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser("run")
    parser.add_argument("dataset", type=str, choices=["schiebinger", "mcfarland", "norman", "sciplex_A549", "sciplex_K562", "sciplex_MCF7"])
    args = parser.parse_args()
    
    dataset = "schiebinger"
    metric = "sqeuclidean"
    data_path= "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/"

    if args.dataset == "schiebinger":
        evaluate("schiebinger", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "mcfarland":
        evaluate("mcfarland", group_by="perturbation", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "norman":
        evaluate("norman", group_by="n_guides", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "sciplex_A549":
        evaluate("sciplex_A549", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "sciplex_K562":
        evaluate("sciplex_K562", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=100)
    elif args.dataset == "sciplex_MCF7":
        evaluate("sciplex_MCF7", group_by="dose_value", rank=False, metric=metric, data_path=data_path, k=100)
