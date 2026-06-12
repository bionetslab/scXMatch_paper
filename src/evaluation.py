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
from evaluation_utils import *


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
                        (sv_vals[j] >= sv_vals[i]) and
                        ((mi_vals[j] > mi_vals[i]) or
                         (ssnr_vals[j] > ssnr_vals[i]) or
                         (sv_vals[j] > sv_vals[i]))):
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


def get_mon_df(xm_results = "../evaluation_results/1_1_monotonicity_scxmatch/", 
               bm_results = "../evaluation_results/1_1_monotonicity_benchmark/", 
               mem_results="../evaluation_results/1_9_monotonicity_memento/", 
               edist_results="../evaluation_results/1_12_monotonicity_edistance", 
               xm_effect_size_results="../evaluation_results/1_6_monotonicity_effect_size"):

    xm_result_dfs = read_xm_monotonicity(xm_results)
    bm_result_dfs = read_benchmarking_monotonicity(bm_results)
    memento_result_df = read_memento_monotonicity(mem_results)
    balanced_edistance_results_df = read_balanced_edistance_monotonicity(edist_results)
    unbalanced_edistance_results_df = read_unbalanced_edistance_monotonicity(edist_results)
    xm_effect_size_results_df = read_effect_size_xm_monotonicity(xm_effect_size_results)    
    
    mon = dict()
    for dataset in bm_result_dfs:
        bm_result_dfs[dataset] = pd.concat([
            xm_result_dfs[dataset], 
            bm_result_dfs[dataset], 
            memento_result_df[dataset],
            balanced_edistance_results_df[dataset],
            unbalanced_edistance_results_df[dataset],
            xm_effect_size_results_df[dataset]
        ], axis=1)

        
        if "bhattacherjee" in dataset or "mcfarland" in dataset:
            bm_result_dfs[dataset] = bm_result_dfs[dataset].sort_index(ascending=False) # 48h < 15d

        else:
            bm_result_dfs[dataset] = bm_result_dfs[dataset].sort_index()
        mon[dataset] = bm_result_dfs[dataset].apply(monotonicity, axis=0)

    mon_df = pd.DataFrame(mon)
    mon_df.drop(["deseq2_500", "edgeR_500", "deseq2_200", "edgeR_200"], axis=0, inplace=True)
    melted_mon = pd.melt(mon_df.reset_index().rename({"index": "metric"}, axis=1), id_vars="metric", var_name="dataset")    
    return bm_result_dfs, mon_df, melted_mon


def get_ssnr_df(monotonicity_result_df, 
                bm_results_within = "../evaluation_results/1_2_SSNR_benchmark", 
                xm_results_within = "../evaluation_results/1_2_SSNR_scxmatch/",
                memento_results_within = "../evaluation_results/1_10_SSNR_memento",
                edist_results_within = "../evaluation_results/1_13_SSNR_edistance/",
                xm_effect_size_results_within = "../evaluation_results/1_7_SSNR_effect_size"
                ):
    xm_results_dfs_within = read_xm_SSNR(xm_results_within)
    bm_results_dfs_within = read_benchmarking_SSNR(bm_results_within)
    memento_results_dfs_within = read_memento_SSNR(memento_results_within)
    balanced_edist_results_dfs_within = read_balanced_edistance_SSNR(edist_results_within)
    unbalanced_edist_results_dfs_within = read_unbalanced_edistance_SSNR(edist_results_within)
    xm_effect_size_dfs_results_within = read_effect_size_xm_SSNR(xm_effect_size_results_within)
    
    all_within = dict()
    for df in monotonicity_result_df:
        conc = pd.concat([bm_results_dfs_within[df], 
                          xm_results_dfs_within[df],
                          memento_results_dfs_within[df],
                          balanced_edist_results_dfs_within[df],
                          unbalanced_edist_results_dfs_within[df],
                          xm_effect_size_dfs_results_within[df]
                          ], 
                         axis=1)
        all_within[df] = conc.sort_values(["split", "test_group"])

    all_within = pd.concat(all_within).astype(float).rename({"p": "scXMatch"}, axis=1)
    all_within = all_within.dropna(axis=1)

    ssnr_dict = dict()
    for dataset in xm_results_dfs_within:
        ssnr_dict[dataset] = dict()
        s0_df = monotonicity_result_df[dataset]
        s_df = all_within.loc[dataset]
        for metric in s0_df.columns:
            if metric in ["deseq2_500", "edgeR_500", "deseq2_200", "edgeR_200"]:
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

                try:
                    ssnr_dict[dataset][metric].append(SSNR(s0, s))   
                except:
                    print(dataset, metric, s0, s)
                    ssnr_dict[dataset][metric].append(np.nan)
            ssnr_dict[dataset][metric] = np.mean(ssnr_dict[dataset][metric])
            
    ssnr_df = pd.DataFrame(ssnr_dict).reset_index().rename({"index": "metric"}, axis=1)
    ssnr_df = ssnr_df.set_index("metric").reset_index().rename({"index": "metric"}, axis=1)
    
    melted_ssnr = pd.melt(ssnr_df, id_vars="metric", var_name="dataset")
    return all_within, ssnr_df, melted_ssnr



def get_sv_df(monotonicity_result_df,
              bm_results_var = "../evaluation_results/1_3_var_benchmark",     
              xm_results_var = "../evaluation_results/1_3_var_scxmatch/",
              memento_results_var = "../evaluation_results/1_11_var_memento/",
              edist_results_var = "../evaluation_results/1_14_var_edistance/",
              xm_effect_size_results_var = "../evaluation_results/1_8_SV_effect_size/",
              metrics=['scXMatch', 'augur', 'wilcoxon', 'deseq2_100', 'edgeR_100', 'memento', 'balanced_edistance', 'unbalanced_edistance', 'effect_size']
              ):
    bm_results_dfs_var = read_benchmark_var(bm_results_var)
    xm_results_dfs_var = read_xm_var(xm_results_var)
    memento_results_dfs_var = read_memento_var(memento_results_var)
    effect_size_var = read_effect_size_var(xm_effect_size_results_var)
    balanced_edist_var = read_balanced_edistance_var(edist_results_var)
    unbalanced_edist_var = read_unbalanced_edistance_var(edist_results_var)    
    
    all_var = dict()
    for dataset in xm_results_dfs_var:
        conc = pd.concat([bm_results_dfs_var[dataset], 
                          xm_results_dfs_var[dataset], 
                          memento_results_dfs_var[dataset],
                          effect_size_var[dataset],
                          balanced_edist_var[dataset],
                          unbalanced_edist_var[dataset]
                          ], axis=1)
        all_var[dataset] = conc
    
    var_res_all = pd.concat(all_var)#.dropna(axis=1)

    var_dict = dict()
    
    for dataset in xm_results_dfs_var:
        var_dict[dataset] = dict()
        split_df = var_res_all.loc[dataset]
        test_groups = split_df.reset_index()["test_group"].unique()
        results = np.zeros((len(test_groups), len(split_df.columns)))
        for i, test_group in enumerate(test_groups):
            results[i] = split_df.loc[test_group].var(axis=0).values
        var_dict[dataset] = np.mean(results, axis=0)
        
        
    for dataset in xm_results_dfs_var:
        var_dict[dataset] = dict()
        s0_df = monotonicity_result_df[dataset]
        s_df = var_res_all.loc[dataset]
        for metric in metrics:
            if metric in ["deseq2_500", "edgeR_500", "deseq2_200", "edgeR_200"]:
                continue
            var_dict[dataset][metric] = list()
            for test_group in s0_df.index:
                s0 = s0_df.loc[test_group, metric]
                if dataset == "schiebinger":
                    if int(test_group) == test_group:
                        s = s_df.loc["D"+str(int(test_group)), metric].values
                    else:
                        s = s_df.loc["D"+str(test_group), metric].values

                else:
                    s = s_df.loc[str(test_group), metric].values
                    
                if s0 == np.nan:
                    raise ValueError(f"s0 is nan for dataset {dataset}, metric {metric}, test_group {test_group}")
                
                if metric == "scXMatch":
                    s0 = 1 - s0
                for si in s:
                    if si != np.nan:
                        di = 1 - np.abs(si - s0) / (np.abs(s0) + 1e-15)
                        var_dict[dataset][metric].append(di)
                    else:
                        raise ValueError(f"si is nan for dataset {dataset}, metric {metric}, test_group {test_group}")
            var_dict[dataset][metric] = np.nanmean(var_dict[dataset][metric])
        
    var_res = pd.DataFrame(var_dict, index=var_res_all.columns)
    melted_var = pd.melt(var_res.reset_index().rename({"index": "metric"}, axis=1), id_vars="metric", var_name="dataset")    
    return var_res_all, var_res, melted_var