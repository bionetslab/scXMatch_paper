import numpy as np
import networkx as nx
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from itertools import chain
from src import *

def draw_ellipse(ax, mean, cov, color):
    ellipse = Ellipse(mean, width=2*cov[0], height=2*cov[1], edgecolor=None, facecolor=color, alpha=0.2)
    ax.add_patch(ellipse)

# Function to compute matching
def get_matching(adata, test, reference):
    distances = calculate_distances_nx(adata.X, "sqeuclidean")
    G = construct_graph_from_distances_nx(distances)
    matching = match_nx(G)
    p_val, z, support = rosenbaum(adata, group_by="Group", test_group=test, reference=reference, use_nx=True)
    return adata, matching, p_val, z, support

# Function to plot matching
def draw_matching(adata, matching, p_val, z, support, ax, group_colors):
    pos = {i: adata.X[i] for i in chain.from_iterable(matching)}
    G = nx.Graph()
    e_colors = [(0.9, 0.9, 0.9) if adata.obs["Group"].iloc[u] == adata.obs["Group"].iloc[v] else (0, 0, 0) for u, v in matching]
    G.add_edges_from(matching)
    
    n_colors = adata.obs["Group"].map(group_colors).values
    nx.draw(G, pos=pos, node_color=n_colors, edge_color=e_colors, node_size=10, ax=ax)
    ax.set_title(f"P={p_val:.5f}, z={z:.3f}")

# Create figure
def legend_elements(groups, group_colors, group_means, group_covs):
    elements = list()
    for group in groups:
        elements += [Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors[group], markersize=8, label=rf'$\mathcal{{N}}({group_means[group]}, {group_covs[group]})$')]

    elements += [
        Line2D([0], [0], color=(0.9, 0.9, 0.9), lw=2, label='Iso-match'),
        Line2D([0], [0], color=(0, 0, 0), lw=2, label='Cross-match')
    ]
    return elements
