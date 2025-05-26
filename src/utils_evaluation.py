import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import os
import scanpy as sc
from statsmodels.stats.multitest import multipletests
import sys
sys.path.append("../src")
from evaluation_metrics import *


import pandas as pd
import numpy as np


def compute_pareto_optimal_datasets(mi: pd.DataFrame, ssnr: pd.DataFrame, sv: pd.DataFrame):
    methods = mi.index.tolist()
    datasets = mi.columns.tolist()
    
    pareto_counts = pd.Series(0, index=methods)
    pareto_optimal_datasets = {method: [] for method in methods}
    
    def is_pareto_optimal(mi_vals, ssnr_vals, sv_vals):
        num_methods = len(mi_vals)
        is_pareto = np.ones(num_methods, dtype=bool)
        for i in range(num_methods):
            for j in range(num_methods):
                if i != j:
                    if ((mi_vals[j] >= mi_vals[i]) and
                        (ssnr_vals[j] >= ssnr_vals[i]) and
                        (sv_vals[j] <= sv_vals[i]) and
                        ((mi_vals[j] > mi_vals[i]) or
                         (ssnr_vals[j] > ssnr_vals[i]) or
                         (sv_vals[j] < sv_vals[i]))):
                        is_pareto[i] = False
                        break
        return is_pareto

    for dataset in datasets:
        mi_vals = mi[dataset].values
        ssnr_vals = ssnr[dataset].values
        sv_vals = sv[dataset].values
        pareto_flags = is_pareto_optimal(mi_vals, ssnr_vals, sv_vals)
        for idx, method in enumerate(methods):
            if pareto_flags[idx]:
                pareto_counts[method] += 1
                pareto_optimal_datasets[method].append(dataset)
    
    return pareto_counts, pareto_optimal_datasets



def discrete_level(x, levels=[1, 0.1, 0.05, 0.01, 0.001]):
    for i, level in enumerate(levels[1:]):
        if x > level:
            return i / (len(levels) - 1) 
    return (i + 1) / (len(levels) - 1)  
        
def get_mon_df(xm_results = "../evaluation_results/1_1_monotonicity_scxmatch/", bm_results = "../evaluation_results/1_1_monotonicity_benchmark/"):
    xm_result_dfs = {f.split("_results.txt")[0]: pd.read_csv(os.path.join(xm_results, f)) for f in os.listdir(xm_results) if f.endswith("results.txt")}
    bm_result_dfs = {f.split("benchmark_results_processed_")[1].split(".")[0]: pd.read_csv(os.path.join(bm_results, f), index_col=0) for f in os.listdir(bm_results) if f.endswith("csv")}
    mon = dict()
    for dataset in bm_result_dfs:
        if "bhattacherjee" in dataset:
            bm_result_dfs[dataset] = bm_result_dfs[dataset].sort_index(ascending=False) # 48h < 15d
        else:
            bm_result_dfs[dataset] = bm_result_dfs[dataset].sort_index()
        bm_result_dfs[dataset]["scXMatch"] = xm_result_dfs[dataset].sort_values("test_group")["p"].values
        # convert the p-values to a discrete level
        # bm_result_dfs[dataset]["discrete scXMatch"] = bm_result_dfs[dataset]["scXMatch"].apply(lambda x: discrete_level(x))
        bm_result_dfs[dataset]["scXMatch"] = 1 - bm_result_dfs[dataset]["scXMatch"]

        bm_result_dfs[dataset][['wilcoxon', 'deseq2_100', 'edgeR_100', 'deseq2_200',
        'edgeR_200', 'deseq2_500', 'edgeR_500']] = bm_result_dfs[dataset][['wilcoxon', 'deseq2_100', 'edgeR_100', 'deseq2_200',
        'edgeR_200', 'deseq2_500', 'edgeR_500']].astype(float) / 2000

        mon[dataset] = bm_result_dfs[dataset].apply(monotonicity, axis=0)

    mon_df = pd.DataFrame(mon)
    mon_df.drop(["deseq2_500", "edgeR_500", "deseq2_200", "edgeR_200"], axis=0, inplace=True)
    melted_mon = pd.melt(mon_df.reset_index().rename({"index": "metric"}, axis=1), id_vars="metric", var_name="dataset")
    melted_mon.replace({"deseq2_100": "#DEGs $DESeq2_{100}$", 
                   "deseq2_200": "#DEGs $DESeq2_{200}$",
                   "edgeR_100": "#DEGs $edgeR_{100}$",
                   "edgeR_200": "#DEGs $edgeR_{200}$",
                  "augur": "$Augur$",
                    "wilcoxon": "#DEGs $Wilcoxon$",
                   "scXMatch": "$\mathbf{scXMatch}$",
                   #"discrete scXMatch": "$discrete~\mathbf{scXMatch}$"
                   }, inplace=True)
    return bm_result_dfs, mon_df, melted_mon


def get_ssnr_df(bm_result_dfs, bm_results_within = "../evaluation_results/1_2_SSNR_benchmark", xm_results_within = "../evaluation_results/1_2_SSNR_scxmatch/"):
    bm_results_dfs_within = {f.split("_aggregated.csv")[0]: pd.read_csv(os.path.join(bm_results_within, f)).drop("Unnamed: 0", axis=1) for f in os.listdir(bm_results_within) if "csv" in f}
    xm_results_dfs_within = {f.split("within_processed_")[1].split("_results.txt")[0]: pd.read_csv(os.path.join(xm_results_within, f)) for f in os.listdir(xm_results_within) if f.endswith("txt")}
    
    all_within = dict()
    for df in xm_results_dfs_within:
        bm_results_dfs_within[df].rename({"split_1": "split"}, inplace=True, axis=1)
        bm_results_dfs_within[df] = bm_results_dfs_within[df].astype(str)
        bm_results_dfs_within[df] = bm_results_dfs_within[df][bm_results_dfs_within[df]["split"].isin(["30", "50", "10"])]
        bm_results_dfs_within[df].set_index(["test_group", "split"], inplace=True)

        xm_results_dfs_within[df] = xm_results_dfs_within[df][~xm_results_dfs_within[df]["group_by"].isin(["split_1", "split_20", "split_40"])]
        xm_results_dfs_within[df] = xm_results_dfs_within[df].astype(str)
        xm_results_dfs_within[df]["group_by"] = xm_results_dfs_within[df]["group_by"].apply(lambda x: x.split("_")[1])
        xm_results_dfs_within[df].rename({"group_by": "split"}, inplace=True, axis=1)
        xm_results_dfs_within[df].set_index(["test_group", "split"], inplace=True)
        xm_results_dfs_within[df].drop(["k", "z", "s", "N"], axis=1, inplace=True)

        conc = pd.concat([bm_results_dfs_within[df], xm_results_dfs_within[df]], axis=1)
        conc["p"] = 1 - conc["p"].astype(float)
        conc[["wilcoxon", "deseq2_100", "edgeR_100", "deseq2_200", "edgeR_200"]] = conc[["wilcoxon", "deseq2_100", "edgeR_100", "deseq2_200", "edgeR_200"]].astype(float) / 2000
        all_within[df] = conc.sort_values(["split", "test_group"])
        
    all_within = pd.concat(all_within).astype(float).rename({"p": "scXMatch"}, axis=1)
    
    ssnr_dict = dict()
    for dataset in xm_results_dfs_within:
        ssnr_dict[dataset] = dict()
        s0_df = bm_result_dfs[dataset]
        s_df = all_within.loc[dataset]
        for metric in s0_df.columns:
            if metric in ["deseq2_500", "edgeR_500"]:
                continue
            ssnr_dict[dataset][metric] = list()
            for test_group in s0_df.index:
                s0 = s0_df.loc[test_group, metric]
                if dataset == "schiebinger":
                    if int(test_group) == test_group:
                        s = s_df.loc["D"+str(int(test_group)), metric].values
                    else:
                        s = s_df.loc["D"+str(test_group), metric].values

                else:
                    s = s_df.loc[str(test_group), metric].values

                ssnr_dict[dataset][metric].append(SSNR(s0, s))
            ssnr_dict[dataset][metric] = np.mean(ssnr_dict[dataset][metric])
    ssnr_df = pd.DataFrame(ssnr_dict).reset_index().rename({"index": "metric"}, axis=1)
    ssnr_df = ssnr_df.set_index("metric").drop(["deseq2_200", "edgeR_200"], axis=0).reset_index().rename({"index": "metric"}, axis=1)
    
    
    melted_ssnr = pd.melt(ssnr_df, id_vars="metric", var_name="dataset")
    melted_ssnr.replace({"deseq2_100": "#DEGs $DESeq2_{100}$", 
                "deseq2_200": "#DEGs $DESeq2_{200}$",
                "edgeR_100": "#DEGs $edgeR_{100}$",
                "edgeR_200": "#DEGs $edgeR_{200}$",
                "augur": "$Augur$",
                "wilcoxon": "#DEGs $Wilcoxon$",
                "scXMatch": "$\mathbf{scXMatch}$",
              #     "discrete scXMatch": "$discrete~\mathbf{scXMatch}$"
            }, inplace=True)
    
    return all_within, ssnr_df, melted_ssnr



def get_sv_df(bm_results_var = "../evaluation_results/1_3_var_benchmark",     xm_results_var = "../evaluation_results/1_3_var_scxmatch/"):
    bm_results_dfs_var = {f.split("_aggregated.csv")[0]: pd.read_csv(os.path.join(bm_results_var, f)) for f in os.listdir(bm_results_var) if "csv" in f}
    xm_results_dfs_var = {f.split("processed_")[1].split("_results.txt")[0]: pd.read_csv(os.path.join(xm_results_var, f)) for f in os.listdir(xm_results_var) if f.endswith("txt")}
    
    all_var = dict()
    for dataset in xm_results_dfs_var:
        #bm_results_dfs_var[dataset] = 
        xm_results_dfs_var[dataset] = xm_results_dfs_var[dataset][xm_results_dfs_var[dataset]["group_by_split"].isin(["split_10", "split_30", "split_50"]) & xm_results_dfs_var[dataset]["group_by_split_reference"].isin(["split_10", "split_30", "split_50"])]
        xm_results_dfs_var[dataset].replace({"split_10": 10, "split_30": 30, "split_50": 50}, inplace=True)
        xm_results_dfs_var[dataset].rename({"group_by_split": "split_1", "group_by_split_reference": "split_2", "p":"scXMatch"}, inplace=True, axis=1)
        xm_results_dfs_var[dataset].set_index(["test_group", "split_1", "split_2"], inplace=True)
        xm_results_dfs_var[dataset].drop(["k", "z", "s"], inplace=True, axis=1)

        bm_results_dfs_var[dataset] = bm_results_dfs_var[dataset].rename({"Unnamed: 0": "test_group"}, axis=1)
        bm_results_dfs_var[dataset] = bm_results_dfs_var[dataset][bm_results_dfs_var[dataset]["split_1"].isin([10, 30, 50]) & bm_results_dfs_var[dataset]["split_2"].isin([10, 30, 50])]
        bm_results_dfs_var[dataset] = bm_results_dfs_var[dataset].set_index(["test_group", "split_1", "split_2"])
        conc = pd.concat([bm_results_dfs_var[dataset], xm_results_dfs_var[dataset]], axis=1)
        all_var[dataset] = conc
    
    var_res_all = pd.concat(all_var).dropna(axis=1)
    #var_res_all["discrete scXMatch"] = var_res_all["scXMatch"].astype(float).apply(lambda x: discrete_level(x))
    var_res_all["scXMatch"] = 1 - var_res_all["scXMatch"].astype(float)

    var_res_all[["wilcoxon", "deseq2_100", "edgeR_100", "deseq2_200", "edgeR_200"]] = var_res_all[["wilcoxon", "deseq2_100", "edgeR_100", "deseq2_200", "edgeR_200"]].astype(float) / 2000
    var_dict = dict()
    
    for dataset in xm_results_dfs_var:
        var_dict[dataset] = dict()
        split_df = var_res_all.loc[dataset]
        test_groups = split_df.reset_index()["test_group"].unique()
        results = np.zeros((len(test_groups), len(split_df.columns)))
        for i, test_group in enumerate(test_groups):
            results[i] = split_df.loc[test_group].var(axis=0).values
        var_dict[dataset] = np.mean(results, axis=0)
        
    var_res = pd.DataFrame(var_dict, index=var_res_all.columns)
    var_res = var_res.drop(["deseq2_200", "edgeR_200"])
    melted_var = pd.melt(var_res.reset_index().rename({"index": "metric"}, axis=1), id_vars="metric", var_name="dataset")
    melted_var.replace({"deseq2_100": "#DEGs $DESeq2_{100}$", 
                   "deseq2_200": "#DEGs $DESeq2_{200}$",
                   "edgeR_100": "#DEGs $edgeR_{100}$",
                   "edgeR_200": "#DEGs $edgeR_{200}$",
                  "augur": "$Augur$",
                    "wilcoxon": "#DEGs $Wilcoxon$",
                   "scXMatch": "$\mathbf{scXMatch}$",
                   #"discrete scXMatch": "$discrete~\mathbf{scXMatch}$"
                   }, inplace=True)
    
    return var_res_all, var_res, melted_var