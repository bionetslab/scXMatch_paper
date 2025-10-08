import pertpy as pt
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys


def get_DEGs(result_df, alpha=0.05, bonferroni=True):
    if bonferroni:
        alpha /= len(result_df)
    return np.sum(result_df["p_value"] < 0.05)


def augur_scores(adata, group_by, reference):
    groups = adata.obs[group_by].unique()
    #adata.obs["cell_type"] = "dummy_cell_type"
    augur_results = dict()
    for test_group in groups:
        if test_group == reference:
            continue
        subset = adata[adata.obs[group_by].isin([reference, test_group])]
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
    try:
        print("calculating augur scores")
        augur_results = augur_scores(adata, group_by, reference)
    except:
        print("augur failed")
        augur_results = dict()
       
    try:
        print("calculating wilcoxon scores")
        wilcoxon_results = wilcoxon(adata, group_by, reference)
    except:
        print("wilcoxon failed")
        wilcoxon_results = dict()
        
    try:
        print("calculating pseudo-bulk data")
        pdata_100 = get_pseudo_bulk_data(adata, group_by, pseudo_bulk="pseudo_bulk_100")
        try:
            print("calculating deseq2 scores")
            deseq2_results_100 = deseq2(pdata_100, group_by, reference)
        except:
            deseq2_results_100 = dict()
            
        try:
            print("calculating edgeR scores")
            edgeR_results_100 = edgeR(pdata_100, group_by, reference)
        except:
            edgeR_results_100 = dict()  
            
    except:
        print("pseudo-bulk failed")
        pdata_100 = None
        deseq2_results_100 = dict()
        edgeR_results_100 = dict()
    
    try:
        print("calculating pseudo-bulk data")
        pdata_200 = get_pseudo_bulk_data(adata, group_by, pseudo_bulk="pseudo_bulk_200")
        try:
            print("calculating deseq2 scores")
            deseq2_results_200 = deseq2(pdata_200, group_by, reference)
        except:
            deseq2_results_200 = dict()
            
        try:
            print("calculating edgeR scores")
            edgeR_results_200 = edgeR(pdata_200, group_by, reference)
        except:
            edgeR_results_200 = dict()  
            
    except:
        print("pseudo-bulk failed")
        pdata_200 = None
        deseq2_results_200 = dict()
        edgeR_results_200 = dict()
    
    try:
        print("calculating pseudo-bulk data")
        pdata_500 = get_pseudo_bulk_data(adata, group_by, pseudo_bulk="pseudo_bulk_500")
        try:
            print("calculating deseq2 scores")
            deseq2_results_500 = deseq2(pdata_500, group_by, reference)
        except:
            deseq2_results_500 = dict()
            
        try:
            print("calculating edgeR scores")
            edgeR_results_500 = edgeR(pdata_500, group_by, reference)
        except:
            edgeR_results_500 = dict()  
            
    except:
        print("pseudo-bulk failed")
        pdata_500 = None
        deseq2_results_500 = dict()
        edgeR_results_500 = dict()
       
      
    return {
        "augur": augur_results,
        "wilcoxon": wilcoxon_results,
        "deseq2_100": deseq2_results_100,
        "edgeR_100": edgeR_results_100,
        "deseq2_200": deseq2_results_200,
        "edgeR_200": edgeR_results_200,
        "deseq2_500": deseq2_results_500,
        "edgeR_500": edgeR_results_500
    }


def main(dataset_path):
    names = [f for f in os.listdir(dataset_path) if (f.endswith("hdf5") and f.startswith("processed") and ("bhatta" in f))]
    files = [os.path.join(dataset_path, f) for f in names]
    print(names)
    print(files)
    
    for f in files:
        adata = ad.read_h5ad(f)
        if "mcfarland" in f:
            group_by = "perturbation_grouped"
            reference = "control"
            
        elif "norman" in f:
            group_by = "n_guides"
            reference = "control"
            
        elif "sciplex" in f:
            group_by = "dose_value"
            reference = "0.0"
            
        elif "schiebinger" in f:
            group_by = "perturbation"
            reference = "control"
            
        elif "bhatta" in f:
            group_by = "label"
            reference = "Maintenance_Cocaine"
            
        else:
            raise ValueError("Unknown dataset")
        
        print("reading", f)
        adata = ad.read_h5ad(f)
        adata.obs[group_by] = adata.obs[group_by].astype(str)
        print(adata.obs[group_by].value_counts(), file=sys.stderr)
        
        results = benchmark_all(adata, group_by, reference)
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv(f"{dataset_path}/benchmark_results_augur_{os.path.basename(f)}.csv", index=True)

            
if __name__ == "__main__":
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji")