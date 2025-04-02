import numpy as np
import itertools
import logging
import time
import sys
import psutil
import pandas as pd
import anndata as ad
sys.path.append("../../scxmatch/src")
from scxmatch import *


a = ad.read_h5ad("/data_nfs/yb85avem/anna/adata.h5ad")

a = scanpy_setup(a)
#before -> a.obs["date"] == "15072024"]
#during -> a.obs["date"] == "15102024"]
#after -> [a.obs["date"] == "28112024"]

a = a[a.obs["date"].isin([28112024, 15072024])]
print(a)
ps = list()
zs = list()
ms = list()
ns = list()
cell_types = list()
for cell_type in a.obs["cell_type"].unique():
    print(cell_type)
    subset = a[a.obs["cell_type"] == cell_type]

    try:
        p, z, s = rosenbaum(subset, group_by="date", test_group=15072024, reference=28112024, metric="sqeuclidean", rank=False, k=None, return_matching=False)
        m, n = subset.obs["date"].value_counts().values
        if not np.isnan(float(z)):
            cell_types.append(cell_type)
            ps.append(p)
            zs.append(z)
            ms.append(m)
            ns.append(n)
    except ValueError as e:
        print(cell_type)
        print(e)

results = pd.DataFrame(np.array([cell_types, ps, zs, ms, ns]).T, columns=["Cell type", "P", "Z", "m", "n"])
results["Z"] = pd.to_numeric(results["Z"])
results.sort_values(by=["Z"], axis=0).to_csv("nicolai.csv")
