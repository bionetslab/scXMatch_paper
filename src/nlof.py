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
from itertools import chain
from scxmatch import test



def main():
    k = 100
    
    reference = "control"
    group_by = "n_guides"

    data_path = "/home/woody/iwbn/iwbn007h/data/scrnaseq_ji"
    dataset = "processed_norman.hdf5"
    
    adata = read_h5ad(os.path.join(data_path, dataset))    
    groups = adata.obs[group_by].unique()
        
    for k in [10, 20, 50, 100, 200]:
        for test_group in groups:
            if test_group == reference:
                continue
            print(test_group)
            subset = adata[adata.obs[group_by].isin([reference, test_group])].copy()
            p, z, s = test(subset, group_by=group_by, reference=reference, test_group=test_group, k=k)
        
            matching = subset.obs[f'XMatch_partner_{test_group}_vs_{reference}'].dropna().reset_index().values.astype(str)
            used_elements = sorted(list(chain.from_iterable(matching)))
            
            distances = cdist(subset.X.todense(), subset.X.todense(), metric="sqeuclidean")
            lof = LocalOutlierFactor(n_neighbors=k, algorithm='auto', leaf_size=30, metric='precomputed')
            pred = lof.fit_predict(np.abs(distances))
            nof = lof.negative_outlier_factor_
            outliers = pd.DataFrame(nof, columns=["Negative outlier factor"], index=subset.obs_names)
            outliers["in matching"] = False
            outliers.loc[used_elements, "in matching"] = True

            outliers.to_csv(f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/3_nlof/{dataset}_{test_group}_{k}_{s:.5f}_outlier.csv")
        
    
if __name__ == "__main__":
    main()