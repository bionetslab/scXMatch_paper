import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
import time
from memento import binary_test_1d, ht_1d_moments
import memento
from scipy.sparse import csr_matrix, issparse


def get_memento_log(adata, treatment_col="stim", reference=0):
    rows = list()
    adata.X = csr_matrix(adata.X)
    for test_group in adata.obs[treatment_col].unique():
        if test_group != reference:
            print(f"Running Memento for {test_group} vs {reference}...")
            t3 = time.time()
            subset = adata[adata.obs[treatment_col].isin([test_group, reference])].copy()
            num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            result = binary_test_1d(
                adata=subset, 
                capture_rate=0.07, # chromium single cell  
                treatment_col=treatment_col, 
                num_cpus=num_cpus,
                num_boot=5000)
            t4 = time.time()
            rows.append({"test_group": test_group, "reference": reference, "#DEGs": len(result.query('de_pval < 0.05')), "time": t4-t3})
    return pd.DataFrame(rows)


def get_memento_log_with_replicate_col(adata, treatment_col="stim", covariate_col="replicate", reference=0):
    rows = list()

    
    for test_group in adata.obs[treatment_col].unique():
        if test_group != reference:
            print(f"Running Memento for {test_group} vs {reference}...")
            t3 = time.time()
            subset = adata[adata.obs[treatment_col].isin([test_group, reference])].copy()
            subset.obs['capture_rate'] = 0.07
            memento.setup_memento(subset, q_column='capture_rate')
            memento.create_groups(subset, label_columns=[treatment_col, covariate_col])
            memento.compute_1d_moments(subset, min_perc_group=.9)
            sample_meta = memento.get_groups(subset)
            sample_meta[covariate_col] = sample_meta[covariate_col].astype('category')
            treatment_df = sample_meta[[treatment_col]]
            cov_df = pd.get_dummies(sample_meta[covariate_col].astype('category'))

            num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            memento.ht_1d_moments(
                adata=subset, 
                treatment=treatment_df,
                covariate=cov_df, 
                num_cpus=num_cpus,
                num_boot=5000)
            result_1d = memento.get_1d_ht_result(subset)
            print(result_1d)
            t4 = time.time()
            rows.append({"test_group": test_group, "reference": reference, "#DEGs": len(result_1d.query('de_pval < 0.05')), "time": t4-t3})
    return pd.DataFrame(rows)


def main(dataset_path):
    name = sys.argv[1]
    names = sorted([f for f in os.listdir(dataset_path) if ((f.endswith("hdf5") or f.endswith("h5ad")) and (name in f))]) #  ]
    files = [os.path.join(dataset_path, f) for f in names]
    print("DATASET FILES", files)
    
    for f in files:
        basen = os.path.basename(f)
        p = f"/home/woody/iwbn/iwbn007h/scXMatch_paper/evaluation_results/1_9_monotonicity_memento/memento_benchmark_results_{basen}.csv"
        if os.path.exists(p):
            print(f"skipping {p}, results already exist.", flush=True, file=sys.stderr)
            continue
        else:
            print(f"Processing {p}", file=sys.stderr)
            
        adata = ad.read_h5ad(f)

        if "mcfarland" in f:
            mcfarland_idxs = [
                ("BICR31", "Idasanutlin"),
                ("BICR31", "Trametinib"),
                ("CAL62", "BRD3379"),
                ("IGROV1", "BRD3379"),
                ("OAW42", "BRD3379")]
            datasets = dict()
            for i, (cell_line, perturbation) in enumerate(mcfarland_idxs):
                if os.path.exists(p.replace(".csv", f"memento_{i}.csv")):
                    continue
                subset = adata[(adata.obs["cell_line"] == cell_line) & (adata.obs["perturbation"].isin([perturbation, "control"]))].copy()

                subset.obs['pert_time'] = np.where(subset.obs['perturbation'] == "control", 0, subset.obs['time'])
                subset.obs['stim'] = subset.obs['pert_time'].replace({"control": 0, "6": 6, "24": 24})
                subset = subset[subset.obs['stim'].isin([0, 6, 24])].copy()
                subset.X = csr_matrix(subset.X)
                datasets[i] = subset
                        
            group_by = "stim"
            reference = 0
                
        elif "norman" in f:
            datasets = dict()
            group_by = "stim"
            reference = 0
            subset = adata[adata.obs['stim'].isin([0, 1, 2])].copy()
            subset.X = subset.layers['counts'].astype(np.int32)
            subset.X = csr_matrix(subset.X)
            datasets[0] = subset

        elif "schiebinger" in f:
            group_by = "stim"
            reference = 0
            covariate_col = "replicate"
            adata.obs["stim"] = np.where(adata.obs["perturbation"] == "control", 0, adata.obs["perturbation"].apply(lambda x: x[1:]))
            adata.obs["stim"] = adata.obs["stim"].astype(float)
            datasets = {0: adata}
            
        elif "bhatta" in f:
            group_by = "label"
            reference = "Maintenance_Cocaine"
                
            bhattacherjee_idxs = [
                "Astro",
                "Endo",
                "Excitatory"]
            
            datasets = dict()
            for i, cell_type in enumerate(bhattacherjee_idxs):
                if os.path.exists(p.replace(".csv", f"memento_{i}.csv")):
                    continue
                subset = adata[(adata.obs["cell_type"] == cell_type)].copy()
                print(subset)
                subset.obs['stim'] = subset.obs['label'].replace({"Maintenance_Cocaine": 0, "withdraw_48h_Cocaine": 1, "withdraw_15d_Cocaine": 2})
                subset = subset[subset.obs['stim'].isin([0, 1, 2])].copy()
                subset.X = csr_matrix(subset.X)
                datasets[i] = subset
                        
            group_by = "stim"
            reference = 0
            covariate_col = "replicate"
            
        else:
            raise ValueError("Unknown dataset")
        
        print(datasets)
        for i in datasets:
            print(f"Running Memento for dataset {i}...")
            print(datasets[i].obs[group_by].value_counts())
            results_df = get_memento_log(datasets[i], group_by, reference)
            results_df.to_csv(p.replace(".csv", f"memento_{i}.csv"))
        
if __name__ == "__main__":
    main("/home/woody/iwbn/iwbn007h/data/scrnaseq_ji/raw")