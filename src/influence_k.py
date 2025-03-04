import scanpy as sc
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
from anndata import read_h5ad
import seaborn as sns
import sys
sys.path.append("..")
from src import *
np.random.seed(42)
sc._settings.ScanpyConfig.n_jobs = -1

def subsample_adata(adata, group_by):
    min_count = adata.obs[group_by].value_counts().min()
    adata_subsampled = adata.copy()
    sampled_indices = []
    
    for dose in adata.obs[group_by].unique():
        group_indices = adata.obs[adata.obs[group_by] == dose].index
        sampled_group_indices = np.random.choice(group_indices, min_count, replace=False)
        sampled_indices.extend(sampled_group_indices)
    
    adata_subsampled = adata_subsampled[sampled_indices, :]
    return adata_subsampled


def jaccard_similarity(matching1, matching2):
    """
    Computes the Jaccard similarity between two graphs given their edge lists.
    """
    edges1 = set(matching1)
    edges2 = set(matching2)
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    return intersection / union if union != 0 else 0


def perfect_match_percentage(matching, reference_matching):
    """
    Computes the Jaccard similarity between two graphs given their edge lists.
    """
    edges1 = set(matching)
    edges2 = set(reference_matching)
    intersection = len(edges1 & edges2)
    perfect = len(edges2)
    return intersection / perfect if perfect != 0 else 0



# Define a function to categorize the values
def categorize_perturbation(value):
    if value == "control":
        return "control"
    elif "_6" in value:
        return "t_6"
    elif "_24" in value:
        return "t_24"
    else:
        return "unknown"


def get_result_df(adata, group_by, reference, test_group, ks):
    (ref_p, ref_z, ref_s), ref_G, ref_matching = rosenbaum(adata, group_by=group_by, reference=reference, test_group=test_group, k=None, return_matching=True)
    
    results = dict()
    for k in ks:
        print(k)
        if len(adata[adata.obs[group_by].isin([reference, test_group])]) - k > 500:
            (p, z, s), G, matching = rosenbaum(adata, group_by=group_by, reference=reference, test_group=test_group, k=k, return_matching=True)
            jaccard_sim = jaccard_similarity(ref_matching, matching)
            match_percentage = perfect_match_percentage(matching, ref_matching)
            num_edges = len(G.get_edges())
            results[k] = [p, z, s, jaccard_sim, match_percentage, num_edges]
    
    results[len(ref_G.get_vertices()) - 1] = [ref_p, ref_z, ref_s, 1, 1, len(ref_G.get_edges())]
    result = pd.DataFrame(results, index=["p-val", "z-score", "relative support", "Jaccard similarity to reference matching", "Percentage of reference matching edges in matching", "Number of edges"])
    return result

    
def main(data_path = "/mnt/data", dataset='mcfarland.hdf5'):
    adata = read_h5ad(os.path.join(data_path, dataset))
    

    if "mcfarland" in dataset:
        adata.obs["time_point"] = adata.obs["perturbation"].apply(categorize_perturbation)
        group_by = "time_point"
        reference = "control"
        name = "mcfarland"
    elif "sciplex_MCF7" in dataset:
        group_by = "dose_value"
        reference = 0.0
        name = "sciplex_MCF7"
    elif "sciplex_A549" in dataset:
        group_by = "dose_value"
        reference = 0.0
        name = "sciplex_A549"
    elif "sciplex_K562" in dataset:
        group_by = "dose_value"
        reference = 0.0
        name = "sciplex_K562"
    
    print(name)
    print(adata.obs[group_by].value_counts())
    adata = subsample_adata(adata, group_by)
    print(adata.obs[group_by].value_counts())


    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    adata.obs[group_by] = adata.obs[group_by].astype('category')

    groups = list(adata.obs[group_by].unique())
    colors = sns.color_palette("hls", len(groups) + 1)
    pal = {group: colors[i] for i, group in enumerate(groups + [reference])}

    dfs = dict()
    for group in groups:
        if group != reference:
            print(group)
            dfs[group] = get_result_df(adata=adata, group_by=group_by, reference=reference, test_group=group, ks=[5, 10, 20, 50, 100, 500, 1000])
            print(dfs)
    pd.concat(dfs).to_csv(f"../plots/{name}_combined_results_k_influence_k_small.csv")

    plot = False
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False)

        k_values = []
        for df in dfs.values():
            ks = list(df.columns)
            k_values = list(set(list(k_values) + ks))
        k_values = sorted(k_values)

        # Create a single figure with 5 subplots in one row
        metrics = ['p-val', 'z-score', 'relative support', 'Jaccard similarity to reference matching', 'Number of edges']
        colors = {'t_6': 'blue', 't_24': 'red'}
        
        for i, metric in enumerate(metrics):
            x, y = (i + 1) // 3, (i + 1) % 3
            for key, df in dfs.items():
                axes[x, y].plot(df.columns.astype(float), df.loc[metric], 'o-', label=key, color=pal[key])
            if metric in ['p-val', 'Number of edges']:
                axes[x, y].set_yscale('log')
            axes[x, y].set_title(metric)
            axes[x, y].set_xlabel('k')
            axes[x, y].legend()

            axes[x, y].legend(
                loc='best',
                labels=groups,
                title=group_by
            )

            #axes[x, y].set_xscale("log")
            axes[x, y].set_xticks(k_values) 
                
        sc.pp.neighbors(adata, n_pcs=0)
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=group_by, palette=pal, ax=axes[0,0], legend_loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"../plots/{name}_k_influence.pdf")
        
                
if __name__ == "__main__":
    main(data_path = "/mnt/data", dataset='sciplex_MCF7.hdf5')