import pertpy as pt
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset_config import get_config


def get_DEGs(result_df, alpha=0.05, bonferroni=True):
    if bonferroni:
        alpha /= len(result_df)
    return np.sum(result_df["p_value"] < 0.05)


def augur_scores(adata, group_by, reference):
    groups = adata.obs[group_by].unique()
    adata.obs["cell_type"] = "dummy_cell_type"
    augur_results = dict()
    for test_group in groups:
        if test_group == reference:
            continue
        subset = adata[adata.obs[group_by].isin([reference, test_group])].copy()
        if group_by != "label" and "label" in subset.obs.columns:
            subset.obs = subset.obs.drop(columns="label")
        ag_rfc = pt.tl.Augur("random_forest_classifier")
        loaded_data = ag_rfc.load(subset, label_col=group_by, cell_type_col="cell_type")
        _, v_results = ag_rfc.predict(loaded_data, subsample_size=20, n_threads=16, select_variance_features=True, span=1)
        augur_results[test_group] = v_results['summary_metrics'].loc["mean_augur_score"].item()
    return augur_results


def wilcoxon(adata, group_by, reference):
    groups = adata.obs[group_by].unique()
    wil_results = dict()
    for test_group in groups:
        if test_group == reference:
            continue
        t = pt.tools.WilcoxonTest(adata)
        results = t.compare_groups(adata=adata, column=group_by, baseline=reference, groups_to_compare=[test_group])
        DEGs = get_DEGs(results, alpha=0.05)
        wil_results[test_group] = DEGs
    return wil_results


def deseq2(pdata, group_by, reference):
    groups = pdata.obs[group_by].unique()
    deseq2_results = dict()

    for test_group in groups:
        if test_group == reference:
            continue
        print("GROUPBY VALUE COUNTS", pdata.obs[group_by].value_counts())
        t = pt.tools.PyDESeq2(pdata, design=f"~{group_by}")
        results = t.compare_groups(adata=pdata, column=group_by, baseline=reference, groups_to_compare=[test_group])
        DEGs = get_DEGs(results, alpha=0.05)
        deseq2_results[test_group] = DEGs
    return deseq2_results


def edgeR(pdata, group_by, reference):
    groups = pdata.obs[group_by].unique()
    edgr_results = dict()
    for test_group in groups:
        if test_group == reference:
            continue
        edgr = pt.tl.EdgeR(pdata, design=f"~{group_by}")
        edgr.fit()
        results = edgr.compare_groups(adata=pdata, column=group_by, baseline=reference, groups_to_compare=[test_group])
        DEGs = get_DEGs(results, alpha=0.05)
        edgr_results[test_group] = DEGs
    return edgr_results

        
def get_pseudo_bulk_data(adata, group_by, pseudo_bulk):
    ps = pt.tl.PseudobulkSpace()
    pdata = ps.compute(adata, target_col=group_by, groups_col=pseudo_bulk, layer_key="counts", mode="sum")
    print("PDATA")
    print(pdata)
    print(pdata.obs.head(20))
    return pdata


def benchmark_all(adata, group_by, reference):
    """
    Benchmark all methods on the given AnnData object.
    
    Parameters:
        adata: AnnData object
        group_by: str, column in adata.obs to group by (e.g., 'perturbation')
        reference: str, reference group for comparison
        pseudo_bulk: str, column in adata.obs to use for pseudo-bulking
        n_cells: int, number of cells to sample from each group
    
    Returns:
        Dictionary with results from each method
    """
    augur_results = augur_scores(adata, group_by, reference)
    try:
        print("calculating augur scores")
    except:
        print("augur failed")
        augur_results = dict()
    
    try:
        print("calculating wilcoxon scores")
        wilcoxon_results = wilcoxon(adata, group_by, reference)
    except:
        print("wilcoxon failed")
        wilcoxon_results = dict()
        
    print("calculating pseudo-bulk data")
    pdata_100 = get_pseudo_bulk_data(adata, group_by, pseudo_bulk="pseudo_bulk_100")
    deseq2_results_100 = deseq2(pdata_100, group_by, reference)
            
    print("calculating edgeR scores")
    edgeR_results_100 = edgeR(pdata_100, group_by, reference)
        
    return {
        "augur": augur_results,
        "wilcoxon": wilcoxon_results,
        "deseq2_100": deseq2_results_100,
        "edgeR_100": edgeR_results_100,
    }


def main(dataset_path):
    names = [f for f in os.listdir(dataset_path) if (f.endswith("h5ad") and ("norman_" in f))]
    files = [os.path.join(dataset_path, f) for f in names]
    print("processing")
    print(files, flush=True)
    
    
    for f in files:
        basen = os.path.basename(f)
        p = f"{dataset_path}/benchmark_results_{basen}.csv"
        if os.path.exists(p):
            print(f"Skipping {p}, results already exist.", file=sys.stderr)
            continue
        else:
            print(f"Processing {p}", file=sys.stderr)
            
        adata = ad.read_h5ad(f)
        group_by, reference = get_config(f)
        
        adata = ad.read_h5ad(f)
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        
        results = benchmark_all(adata, group_by, reference)
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{dataset_path}/benchmark_results_{os.path.basename(f)}.csv", index=True)

            
if __name__ == "__main__":
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/")