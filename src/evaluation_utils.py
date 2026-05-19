import pandas as pd 
import numpy as np
import os
import sys
sys.path.append("../src")
from evaluation_metrics import *
from evaluation_utils import *


import pandas as pd
import numpy as np

mcfarland_idxs = [
    ("BICR31", "Idasanutlin"),
    ("BICR31", "Trametinib"),
    ("CAL62", "BRD3379"),
    ("IGROV1", "BRD3379"),
    ("OAW42", "BRD3379")]

bhattacherjee_testgroups = {
    0: "Maintenance_Cocaine", 
    1: "withdraw_48h_Cocaine",
    2: "withdraw_15d_Cocaine"}

number_of_genes_in_raw_data = {
    "norman": 19018,
     "schiebinger": 27998,
     "mcfarland": 32738,
     "bhattacherjee": 18469}

# -------- Functions to read in monotonicity results for different metrics --------
def read_memento_monotonicity(mem_results):
    memento_results = {f.split("memento_benchmark_results_")[1].split(".")[0]: pd.read_csv(os.path.join(mem_results, f), index_col=0).set_index("test_group")[["#DEGs"]] for f in os.listdir(mem_results) if (f.endswith("csv") and not f.startswith("with_cov"))}
    for dataset in memento_results:
        memento_results[dataset].rename({"#DEGs": "memento"}, inplace=True, axis=1)
    return memento_results

def read_xm_monotonicity(xm_results):
    xm_result_dfs = {f.split("_results.txt")[0]: pd.read_csv(os.path.join(xm_results, f)).set_index("test_group")[["p"]] for f in os.listdir(xm_results) if f.endswith("results.txt")}
    for dataset in xm_result_dfs:
        xm_result_dfs[dataset].rename({"p": "scXMatch"}, inplace=True, axis=1)
        xm_result_dfs[dataset]["scXMatch"] = - xm_result_dfs[dataset]["scXMatch"]
    return xm_result_dfs

def read_benchmarking_monotonicity(bm_results):
    bm_result_dfs = {f.split("benchmark_results_processed_")[1].split(".")[0]: pd.read_csv(os.path.join(bm_results, f), index_col=0) for f in os.listdir(bm_results) if f.endswith("csv")}
    for dataset in bm_result_dfs:  
        deg_cols = [c for c in bm_result_dfs[dataset].columns if not ("augur" in c)]
    return bm_result_dfs

def read_balanced_edistance_monotonicity(edist_results):
    balanced_edistance_results = {f.split("processed_")[1].split("_balanced")[0]: pd.read_csv(os.path.join(edist_results, f)).rename({"distance": "balanced_edistance", "testgroup": "test_group"}, axis=1)[["balanced_edistance", "test_group"]] for f in os.listdir(edist_results) if (f.endswith("csv") and ("_balanced" in f))}
    for dataset in balanced_edistance_results:
        if dataset == "schiebinger":
            balanced_edistance_results[dataset]["test_group"] = balanced_edistance_results[dataset]["test_group"].apply(lambda x: float(x[1:]) if x.startswith("D") else float(x))
        balanced_edistance_results[dataset].set_index("test_group", inplace=True)
    return balanced_edistance_results

def read_unbalanced_edistance_monotonicity(edist_results):
    unbalanced_edistance_results = {f.split("processed_")[1].split("_unbalanced")[0]: pd.read_csv(os.path.join(edist_results, f)).rename({"distance": "unbalanced_edistance", "testgroup": "test_group"}, axis=1)[["unbalanced_edistance", "test_group"]] for f in os.listdir(edist_results) if (f.endswith("csv") and ("_unbalanced" in f))}
    for dataset in unbalanced_edistance_results:
        if dataset == "schiebinger":
            unbalanced_edistance_results[dataset]["test_group"] = unbalanced_edistance_results[dataset]["test_group"].apply(lambda x: float(x[1:]) if x.startswith("D") else float(x))
        unbalanced_edistance_results[dataset].set_index("test_group", inplace=True)
    return unbalanced_edistance_results

def read_effect_size_xm_monotonicity(xm_effect_size_results):
    xm_effect_size_results = {f.split("results_processed_")[1].split(".hdf5")[0]: pd.read_csv(os.path.join(xm_effect_size_results, f)).rename({"testgroup": "test_group"}, axis=1) for f in os.listdir(xm_effect_size_results) if f.endswith("csv")}
    for dataset in xm_effect_size_results:
        if dataset == "schiebinger":
            xm_effect_size_results[dataset]["test_group"] = xm_effect_size_results[dataset]["test_group"].apply(lambda x: float(x[1:]) if x.startswith("D") else float(x))
        xm_effect_size_results[dataset] = 1 - xm_effect_size_results[dataset].set_index("test_group")[["effect_size"]]
    return xm_effect_size_results

# -------- Functions to read in SSNR results for different metrics --------
def read_memento_SSNR(mem_results_within):
    # TODO divide by max possible #DEGs for SV?
    memento_results = {f.split("memento_benchmark_results_")[1].split(".")[0]: pd.read_csv(os.path.join(mem_results_within, f), index_col=0) for f in os.listdir(mem_results_within) if (f.endswith("csv") and not f.startswith("with_cov"))}
    memento_results_per_dataset = dict()
    
    for dataset in memento_results:
        memento_results[dataset].drop("test_group", axis=1, inplace=True, errors="ignore")
        memento_results[dataset].rename({"#DEGs": "memento", "testgroup": "test_group"}, inplace=True, axis=1)
        
        if "dataset" in memento_results[dataset].columns:
            for dataset_name, df in memento_results[dataset].groupby("dataset"):
                memento_results_per_dataset[dataset_name.split("processed_")[1]] = df[["memento", "group_by", "test_group"]]
        else:
            memento_results_per_dataset[dataset] = memento_results[dataset][["memento", "group_by", "test_group"]]
            
    for dataset in memento_results_per_dataset:
        if "mcfarland" in dataset:
            idx = dataset.split("mcfarland_")[1]
            drug = mcfarland_idxs[int(idx) - 1][1]
            memento_results_per_dataset[dataset]["test_group"] = f"{drug}_" + memento_results_per_dataset[dataset]["test_group"].astype(str)
        elif "bhattacherjee" in dataset:
            memento_results_per_dataset[dataset]["test_group"] = memento_results_per_dataset[dataset]["test_group"].apply(lambda x: bhattacherjee_testgroups[x])
        elif "schiebinger" in dataset:
            memento_results_per_dataset[dataset]["test_group"] = memento_results_per_dataset[dataset]["test_group"].apply(lambda x: f"D{x}" if int(x) != x else f"D{int(x)}")
        memento_results_per_dataset[dataset]["group_by"] = memento_results_per_dataset[dataset]["group_by"].apply(lambda x: x.split("_")[1])
        memento_results_per_dataset[dataset].rename({"group_by": "split"}, inplace=True, axis=1)
        memento_results_per_dataset[dataset].reset_index(drop=True, inplace=True)
        memento_results_per_dataset[dataset]["test_group"] = memento_results_per_dataset[dataset]["test_group"].astype(str)
        memento_results_per_dataset[dataset].set_index(["test_group", "split"], inplace=True)

    return memento_results_per_dataset

def read_xm_SSNR(xm_results_within):
    xm_results_dfs_within = {f.split("within_processed_")[1].split("_results.txt")[0]: pd.read_csv(os.path.join(xm_results_within, f)) for f in os.listdir(xm_results_within) if f.endswith("txt")}
    for df in xm_results_dfs_within:
        xm_results_dfs_within[df] = xm_results_dfs_within[df][~xm_results_dfs_within[df]["group_by"].isin(["split_1", "split_20", "split_40"])]
        xm_results_dfs_within[df] = xm_results_dfs_within[df].astype(str)
        xm_results_dfs_within[df]["group_by"] = xm_results_dfs_within[df]["group_by"].apply(lambda x: x.split("_")[1])
        xm_results_dfs_within[df].rename({"group_by": "split"}, inplace=True, axis=1)
        xm_results_dfs_within[df]["scXMatch"] = - xm_results_dfs_within[df]["p"].astype(float)
        xm_results_dfs_within[df].drop(["k", "z", "s", "N", "p"], axis=1, inplace=True)
        xm_results_dfs_within[df]["test_group"] = xm_results_dfs_within[df]["test_group"].astype(str)
        xm_results_dfs_within[df].set_index(["test_group", "split"], inplace=True)
    return xm_results_dfs_within

def read_benchmarking_SSNR(bm_results_within):
    bm_results_dfs_within = {f.split("_aggregated.csv")[0]: pd.read_csv(os.path.join(bm_results_within, f)).drop("Unnamed: 0", axis=1, errors="ignore") for f in os.listdir(bm_results_within) if "csv" in f}
    for df in bm_results_dfs_within:
        bm_results_dfs_within[df].rename({"split_1": "split"}, inplace=True, axis=1)
        bm_results_dfs_within[df] = bm_results_dfs_within[df].astype(str)
        bm_results_dfs_within[df] = bm_results_dfs_within[df][bm_results_dfs_within[df]["split"].isin(["30", "50", "10"])]
        deg_cols = [c for c in bm_results_dfs_within[df].columns if ((not "p" in c) and (not "augur" in c))]
        bm_results_dfs_within[df]["test_group"] = bm_results_dfs_within[df]["test_group"].astype(str)
        bm_results_dfs_within[df].set_index(["test_group", "split"], inplace=True)
    return bm_results_dfs_within


def read_balanced_edistance_SSNR(edist_results_within):
    balanced_edistance_results = {f.split("processed_")[1].split("_balanced")[0]: pd.read_csv(os.path.join(edist_results_within, f)).rename({"distance": "balanced_edistance", "testgroup": "test_group", "group_by_split": "split"}, axis=1)[["balanced_edistance", "test_group", "split"]] for f in os.listdir(edist_results_within) if (f.endswith("csv") and ("_balanced" in f))}
    for dataset in balanced_edistance_results:
        balanced_edistance_results[dataset].columns = ["balanced_edistance", "test_group_old", "test_group", "split"]
        balanced_edistance_results[dataset].drop("test_group_old", axis=1, inplace=True)
        balanced_edistance_results[dataset]["split"] = balanced_edistance_results[dataset]["split"].apply(lambda x: x.split("_")[1])
        balanced_edistance_results[dataset]["test_group"] = balanced_edistance_results[dataset]["test_group"].astype(str)
        balanced_edistance_results[dataset].set_index(["test_group", "split"], inplace=True)
    return balanced_edistance_results

def read_unbalanced_edistance_SSNR(edist_results_within):
    unbalanced_edistance_results = {f.split("processed_")[1].split("_unbalanced")[0]: pd.read_csv(os.path.join(edist_results_within, f)).rename({"distance": "unbalanced_edistance", "testgroup": "test_group", "group_by_split": "split"}, axis=1)[["unbalanced_edistance", "test_group", "split"]] for f in os.listdir(edist_results_within) if (f.endswith("csv") and ("_unbalanced" in f))}
    for dataset in unbalanced_edistance_results:
        unbalanced_edistance_results[dataset].columns = ["unbalanced_edistance", "test_group_old", "test_group", "split"]
        unbalanced_edistance_results[dataset].drop("test_group_old", axis=1, inplace=True)
        unbalanced_edistance_results[dataset]["split"] = unbalanced_edistance_results[dataset]["split"].apply(lambda x: x.split("_")[1])
        unbalanced_edistance_results[dataset]["test_group"] = unbalanced_edistance_results[dataset]["test_group"].astype(str)
        unbalanced_edistance_results[dataset].set_index(["test_group", "split"], inplace=True)
    return unbalanced_edistance_results

def read_effect_size_xm_SSNR(xm_effect_size_results_within):
    xm_effect_size_results = {f.split("processed_")[1].split("_results.txt")[0]: pd.read_csv(os.path.join(xm_effect_size_results_within, f)) for f in os.listdir(xm_effect_size_results_within) if f.endswith("txt")}
    for dataset in xm_effect_size_results:
        xm_effect_size_results[dataset] = xm_effect_size_results[dataset][["test_group", "effect_size", "group_by"]]
        xm_effect_size_results[dataset].rename({"group_by": "split"}, inplace=True, axis=1)
        xm_effect_size_results[dataset]["effect_size"] = 1 - xm_effect_size_results[dataset]["effect_size"]
        xm_effect_size_results[dataset]["split"] = xm_effect_size_results[dataset]["split"].apply(lambda x: x.split("_")[1])
        xm_effect_size_results[dataset]["test_group"] = xm_effect_size_results[dataset]["test_group"].astype(str)
        xm_effect_size_results[dataset].set_index(["test_group", "split"], inplace=True)
    return xm_effect_size_results

# -------- Functions to read in var results for different metrics --------

def read_benchmark_var(bm_results_var):
    bm_results_dfs_var = {f.split("_aggregated.csv")[0]: pd.read_csv(os.path.join(bm_results_var, f)) for f in os.listdir(bm_results_var) if "_aggregated.csv" in f}
    for dataset in bm_results_dfs_var:
        bm_results_dfs_var[dataset] = bm_results_dfs_var[dataset].rename({"Unnamed: 0": "test_group"}, axis=1)
        bm_results_dfs_var[dataset] = bm_results_dfs_var[dataset][bm_results_dfs_var[dataset]["split_1"].isin([10, 30, 50]) & bm_results_dfs_var[dataset]["split_2"].isin([10, 30, 50])]
        bm_results_dfs_var[dataset]["test_group"] = bm_results_dfs_var[dataset]["test_group"].astype(str)
        bm_results_dfs_var[dataset]["split_1"] = bm_results_dfs_var[dataset]["split_1"].astype(str)
        bm_results_dfs_var[dataset]["split_2"] = bm_results_dfs_var[dataset]["split_2"].astype(str)
        bm_results_dfs_var[dataset] = bm_results_dfs_var[dataset].set_index(["test_group", "split_1", "split_2"])

    return bm_results_dfs_var

def read_xm_var(xm_results_var):
    xm_results_dfs_var = {f.split("processed_")[1].split("_results.txt")[0]: pd.read_csv(os.path.join(xm_results_var, f)) for f in os.listdir(xm_results_var) if f.endswith("txt")}
    for dataset in xm_results_dfs_var:
        xm_results_dfs_var[dataset] = xm_results_dfs_var[dataset][xm_results_dfs_var[dataset]["group_by_split"].isin(["split_10", "split_30", "split_50"]) & xm_results_dfs_var[dataset]["group_by_split_reference"].isin(["split_10", "split_30", "split_50"])]
        xm_results_dfs_var[dataset].replace({"split_10": 10, "split_30": 30, "split_50": 50}, inplace=True)
        xm_results_dfs_var[dataset].rename({"group_by_split": "split_1", "group_by_split_reference": "split_2", "p":"scXMatch"}, inplace=True, axis=1)
        xm_results_dfs_var[dataset]["test_group"] = xm_results_dfs_var[dataset]["test_group"].astype(str)
        xm_results_dfs_var[dataset]["split_1"] = xm_results_dfs_var[dataset]["split_1"].astype(str)
        xm_results_dfs_var[dataset]["split_2"] = xm_results_dfs_var[dataset]["split_2"].astype(str)
        xm_results_dfs_var[dataset].set_index(["test_group", "split_1", "split_2"], inplace=True)
        xm_results_dfs_var[dataset]["scXMatch"] = 1 - xm_results_dfs_var[dataset]["scXMatch"].astype(float)
        xm_results_dfs_var[dataset].drop(["k", "z", "s"], inplace=True, axis=1)
    return xm_results_dfs_var
    
def read_memento_var(memento_results_var):
    memento_results = {f.split("memento_benchmark_results_")[1].split(".")[0]: pd.read_csv(os.path.join(memento_results_var, f), index_col=0) for f in os.listdir(memento_results_var) if (f.endswith("csv") and not f.startswith("with_cov"))}
    memento_results_per_dataset = dict()

    for dataset in memento_results:
        memento_results[dataset].drop("test_group", axis=1, inplace=True, errors="ignore")
        memento_results[dataset].rename({"#DEGs": "memento", "testgroup": "test_group"}, inplace=True, axis=1)

        if "dataset" in memento_results[dataset].columns:
            for dataset_name, df in memento_results[dataset].groupby("dataset"):
                memento_results_per_dataset[dataset_name.split("processed_")[1]] = df[["memento", "group_by_split_1", "group_by_split_2", "test_group"]]
        else:
            memento_results_per_dataset[dataset] = memento_results[dataset][["memento", "group_by_split_1", "group_by_split_2", "test_group"]]

    for dataset in memento_results_per_dataset:
        if "mcfarland" in dataset:
            idx = dataset.split("mcfarland_")[1]
            drug = mcfarland_idxs[int(idx) - 1][1]
            memento_results_per_dataset[dataset]["test_group"] = f"{drug}_" + memento_results_per_dataset[dataset]["test_group"].astype(str)
        elif "bhattacherjee" in dataset:
            memento_results_per_dataset[dataset]["test_group"] = memento_results_per_dataset[dataset]["test_group"].apply(lambda x: bhattacherjee_testgroups[x])
        elif "schiebinger" in dataset:
            memento_results_per_dataset[dataset]["test_group"] = memento_results_per_dataset[dataset]["test_group"].apply(lambda x: f"D{x}" if int(x) != x else f"D{int(x)}")
        
        memento_results_per_dataset[dataset]["group_by_split_1"] = memento_results_per_dataset[dataset]["group_by_split_1"].apply(lambda x: x.split("_")[1])
        memento_results_per_dataset[dataset].rename({"group_by_split_1": "split_1"}, inplace=True, axis=1)
        
        memento_results_per_dataset[dataset]["group_by_split_2"] = memento_results_per_dataset[dataset]["group_by_split_2"].apply(lambda x: x.split("_")[1])
        memento_results_per_dataset[dataset].rename({"group_by_split_2": "split_2"}, inplace=True, axis=1)
        
        memento_results_per_dataset[dataset].reset_index(drop=True, inplace=True)
        memento_results_per_dataset[dataset]["test_group"] = memento_results_per_dataset[dataset]["test_group"].astype(str)
        memento_results_per_dataset[dataset]["split_1"] = memento_results_per_dataset[dataset]["split_1"].astype(str)
        memento_results_per_dataset[dataset]["split_2"] = memento_results_per_dataset[dataset]["split_2"].astype(str)

        memento_results_per_dataset[dataset].set_index(["test_group", "split_1", "split_2"], inplace=True)
    return memento_results_per_dataset


def read_effect_size_var(xm_effect_size_results_var):
    xm_effect_size_results = {f.split("processed_")[1].split("_results.txt")[0]: pd.read_csv(os.path.join(xm_effect_size_results_var, f)) for f in os.listdir(xm_effect_size_results_var) if f.endswith("txt")}
    for dataset in xm_effect_size_results:
        xm_effect_size_results[dataset] = xm_effect_size_results[dataset][["test_group", "effect_size", "split_1", "split_2"]]
        xm_effect_size_results[dataset]["effect_size"] = 1 - xm_effect_size_results[dataset]["effect_size"]
        xm_effect_size_results[dataset]["split_1"] = xm_effect_size_results[dataset]["split_1"].apply(lambda x: x.split("_")[1]).astype(str)
        xm_effect_size_results[dataset]["split_2"] = xm_effect_size_results[dataset]["split_2"].apply(lambda x: x.split("_")[1]).astype(str)
        xm_effect_size_results[dataset]["test_group"] = xm_effect_size_results[dataset]["test_group"].astype(str)
        xm_effect_size_results[dataset].set_index(["test_group", "split_1", "split_2"], inplace=True)
    return xm_effect_size_results


def read_balanced_edistance_var(edist_results_var):
    balanced_edistance_results = {f.split("processed_")[1].split("_balanced")[0]: pd.read_csv(os.path.join(edist_results_var, f)).drop("test_group", axis=1).rename({"distance": "balanced_edistance", "testgroup": "test_group", "group_by_split": "split_1", "group_by_split_reference": "split_2"}, axis=1)[["balanced_edistance", "test_group", "split_1", "split_2"]] for f in os.listdir(edist_results_var) if (f.endswith("csv") and ("_balanced" in f))}
    for dataset in balanced_edistance_results:
        balanced_edistance_results[dataset]["split_1"] = balanced_edistance_results[dataset]["split_1"].apply(lambda x: x.split("_")[1]).astype(str)
        balanced_edistance_results[dataset]["split_2"] = balanced_edistance_results[dataset]["split_2"].apply(lambda x: x.split("_")[1]).astype(str)
        balanced_edistance_results[dataset]["test_group"] = balanced_edistance_results[dataset]["test_group"].astype(str)
        balanced_edistance_results[dataset].set_index(["test_group", "split_1", "split_2"], inplace=True)
    return balanced_edistance_results


def read_unbalanced_edistance_var(edist_results_var):
    unbalanced_edistance_results = {f.split("processed_")[1].split("_unbalanced")[0]: pd.read_csv(os.path.join(edist_results_var, f)).drop("test_group", axis=1).rename({"distance": "unbalanced_edistance", "testgroup": "test_group", "group_by_split": "split_1", "group_by_split_reference": "split_2"}, axis=1)[["unbalanced_edistance", "test_group", "split_1", "split_2"]] for f in os.listdir(edist_results_var) if (f.endswith("csv") and ("_unbalanced" in f))}
    for dataset in unbalanced_edistance_results:
        unbalanced_edistance_results[dataset]["split_1"] = unbalanced_edistance_results[dataset]["split_1"].apply(lambda x: x.split("_")[1]).astype(str)
        unbalanced_edistance_results[dataset]["split_2"] = unbalanced_edistance_results[dataset]["split_2"].apply(lambda x: x.split("_")[1]).astype(str)
        unbalanced_edistance_results[dataset]["test_group"] = unbalanced_edistance_results[dataset]["test_group"].astype(str)
        unbalanced_edistance_results[dataset].set_index(["test_group", "split_1", "split_2"], inplace=True)
    return unbalanced_edistance_results