import pertpy as pt
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns

def get_DEGs(result_df, alpha=0.05, bonferroni=True):
    assert len(result_df) == 2000
    if bonferroni:
        alpha /= 2000
    return np.sum(result_df["p_value"] < 0.05)


def augur_scores(adata, group_by, reference):
    groups = adata.obs[group_by].unique()
    adata.obs["cell_type"] = "dummy_cell_type"
    augur_results = dict()
    for test_group in groups:
        if test_group == reference:
            continue
        subset = adata[adata.obs[group_by].isin([reference, test_group])]
        ag_rfc = pt.tl.Augur("random_forest_classifier")
        loaded_data = ag_rfc.load(subset, label_col=group_by, cell_type_col="cell_type")
        v_adata, v_results = ag_rfc.predict(
        loaded_data, subsample_size=20, n_threads=4, select_variance_features=True, span=1)
        augur_results[test_group] = v_results['summary_metrics'].loc["mean_augur_score"].item()
    return augur_results


def wilcoxon(adata, group_by, reference):
    groups = adata.obs[group_by].unique()
    wil_results = dict()
    for test_group in groups:
        if test_group == reference:
            continue
        t = pt.tools.WilcoxonTest(test)
        results = t.compare_groups(adata=test, column=group_by, baseline=reference, groups_to_compare=[test_group])
        DEGs = get_DEGs(results, alpha=0.05)
        wil_results[test_group] = DEGs
    return wil_results


def deseq2(pdata, group_by, reference, pseudo_bulk):
    deseq2_results = dict()
    for test_group in groups:
        if test_group == reference:
            continue
        t = pt.tools.PyDESeq2(pdata, design=f"~{group_by}")
        results = t.compare_groups(adata=pdata, column=group_by, baseline=reference, groups_to_compare=[test_group])
        DEGs = get_DEGs(results, alpha=0.05)
        deseq2_results[test_group] = DEGs
    return deseq2_results


def edgeR(pdata, group_by, reference, pseudo_bulk):
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


def benchmark_all(adata, group_by, reference, pseudo_bulk, n_cells=2000):
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
    wilcoxon_results = wilcoxon(adata, group_by, reference)
    pdata = get_pseudo_bulk_data(adata, group_by, pseudo_bulk)
    deseq2_results = deseq2(pdata, group_by, reference)
    edgeR_results = edgeR(pdata, group_by, reference)
    
    return {
        "augur": augur_results,
        "wilcoxon": wilcoxon_results,
        "deseq2": deseq2_results,
        "edgeR": edgeR_results
    }
    
    
def main(dataset_path):
    names = [f for f in os.listdir(dataset_path) if f.endswith("hdf5") and f.startswith("processed")]
    files = [os.path.join(dataset_path, f) for f in names]
    
    for f in files:
        adata = ad.read_h5ad(f)
        adata = scanpy_setup(adata)
        
        if "mcfarland" in f:
            group_by = "perturbation_grouped"
            reference = "control"
            adata.obs[group_by] = adata.obs['perturbation'].apply(lambda x: x.split("_")[-1])
            
        elif "norman" in f:
            group_by = "n_guides"
            reference = "control"
            
        elif "sciplex" in f:
            group_by = "dose_value"
            reference = "0.0"
            
        elif "schiebinger" in f:
            group_by = "perturbation"
            reference = "control"
            
        else:
            raise ValueError("Unknown dataset")
        
        adata = ad.read_h5ad(f)
        augur_results = augur_scores(adata, group_by, reference)
        wilcoxon_results = wilcoxon(adata, group_by, reference)
        pdata = get_pseudo_bulk_data(adata, group_by, pseudo_bulk)
        deseq2_results = deseq2(pdata, group_by, reference)
        edgeR_results = edgeR(pdata, group_by, reference)
        results = {
            "augur": augur_results,
            "wilcoxon": wilcoxon_results,
            "deseq2": deseq2_results,
            "edgeR": edgeR_results
        }
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{dataset_path}/benchmark_results_{os.path.basename(f)}.csv", index=False)

            
if __name__ == "__main__":
    main("datapath")