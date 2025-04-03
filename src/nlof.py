from anndata import read_h5ad
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import LocalOutlierFactor
import string

import sys
sys.path.append("../../scxmatch/src")
from scxmatch import *


def main():
    k = 100
    reference = 0.0
    group_by = "dose_value"

    outlier_dfs = list()
    data_path = "/data_nfs/datasets/scrnaseq_ji/"
    datasets = ["sciplex_A549.hdf5", "sciplex_MCF7.hdf5", "sciplex_K562.hdf5"]
    for i, dataset in enumerate(datasets): 
        adata = read_h5ad(os.path.join(data_path, dataset))    
        adata = scanpy_setup(adata)
    
        groups = adata.obs[group_by].unique()
        #reference_indices = list(np.random.choice(np.where(adata.obs[group_by] == reference)[0], size=n, replace=False))
        
        dfs = dict()
        for test_group in groups:
            if test_group == reference:
                continue
            
            #test_group_indices = np.where(adata.obs[group_by] == test_group)[0]
            #sampled_indices = list(np.random.choice(test_group_indices, size=n, replace=False))
            
            subset = adata[adata.obs[group_by].isin([reference, test_group])]
            (p, z, s), G, matching = rosenbaum(subset, group_by=group_by, reference=reference, test_group=test_group, k=k, return_matching=True)

            used_elements = sorted(list(chain.from_iterable(matching)))
            distances = cdist(subset.X.todense(), subset.X.todense(), metric="sqeuclidean")
            if np.min(distances) < 0:
                print("problem")
            
            lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='precomputed')
            pred = lof.fit_predict(np.abs(distances))
            nof = lof.negative_outlier_factor_
            outliers = pd.DataFrame(nof, columns=["Negative outlier factor"])
            outliers["in matching"] = False
            outliers.iloc[used_elements, 1] = True
            dfs[test_group] = outliers
        df = pd.concat(dfs).reset_index().rename({"level_0": "Test group"}, axis=1)
        df.to_csv(f"{dataset}_outlier.csv")
        outlier_dfs.append(df)

    p_vals = list()
    for i, df in enumerate(outlier_dfs):
        for j, test_group in enumerate(df["Test group"].unique()):
            subset_true = df[(df["Test group"] == test_group) & (df["in matching"] == True)]["Negative outlier factor"]
            subset_false = df[(df["Test group"] == test_group) & (df["in matching"] == False)]["Negative outlier factor"]
            p_val = mannwhitneyu(subset_false, subset_true, alternative="less").pvalue
            p_vals.append(p_val)

    p_adjs = multipletests(p_vals, method="fdr_bh")[1]

    f, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    
    labels = string.ascii_lowercase  
    j_steps = int(len(p_adjs) / len(outlier_dfs))

    for i, df in enumerate(outlier_dfs):
        legend = False if i != 1 else "full"
        sns.violinplot(df, y="Negative local outlier factor", hue="in matching", x="Test group", cut=0, split=True, palette="magma", ax=axs[i], legend=legend) 
        axs[i].set_title(os.path.splitext(datasets[i])[0])
        
        for j in range(0, j_steps):
            axs[i].text(j, -1, f"${p_adjs[i * j_steps + j]:.3f}$", ha="center", va="bottom", fontsize=10, color="black")
    
        axs[i].text(
            -0.05, 1.05,  # Position (normalized figure coordinates)
            labels[i],   # Corresponding letter
            transform=axs[i].transAxes,  # Relative to subplot
            fontsize=10, fontweight='bold', va='top', ha='left'
        )
    
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, title="Sample contained in matching with $k=100$")
    plt.savefig("../plots/nof.pdf", bbox_inches="tight")
    plt.savefig("../plots/nof.svg", bbox_inches="tight")
    

if __name__ == "__main__":
    main()