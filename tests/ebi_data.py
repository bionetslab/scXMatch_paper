import scanpy as sc
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from src import rosenbaum


def test_on_ebi_data(accession="E-MTAB-9221", group_by="Factor Value[clinical history]", test_group="mild COVID-19", reference="Control", metric="euclidean", rank=False):
    adata = sc.datasets.ebi_expression_atlas(accession=accession)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize each cell to have 10,000 total counts
    sc.pp.log1p(adata)  # Logarithmize the data
    sc.pp.highly_variable_genes(adata, n_top_genes=100)  # Select top 2000 variable genes
    adata = adata[:, adata.var.highly_variable]  # Subset to variable genes
    sc.pp.scale(adata, max_value=10)  # Scale each gene to unit variance and clip to max_value=10
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=30)  # Set n_neighbors and use significant PCs
    
    print(rosenbaum(adata, group_by=group_by, test_group=test_group, reference=reference, metric=metric, rank=rank))
    
    sc.tl.umap(adata, min_dist=0.3)  # Set min_dist
    sc.pl.umap(adata, color=group_by)  # Replace 'louvain' with desired cluster or metadata field
    plt.savefig(f"{accession}_umap.pdf")


if __name__ == "__main__":
    test_on_ebi_data()