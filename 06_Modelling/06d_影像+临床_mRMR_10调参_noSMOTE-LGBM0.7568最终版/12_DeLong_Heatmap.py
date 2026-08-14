# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(SCRIPT_DIR, '11_DeLong_Pvalues_FDRcorrected.xlsx')

# ======================== read ========================
df = pd.read_excel(INPUT_FILE, index_col=0)
model_names = list(df.columns)
pval_matrix = df.values
N = len(model_names)
print(f'Models: {N}')
print(model_names)

# ======================== reorder by correlation ========================
# Hierarchical clustering on each model's DeLong p-value profile
# (correlation distance, average linkage) so statistically similar
# models are grouped together -- same idea as 01_correlation_clustermap.
linkage_matrix = linkage(pdist(pval_matrix, metric='correlation'), method='average')
ordered_names = [model_names[i] for i in leaves_list(linkage_matrix)]
print('Reordered (by correlation):', ordered_names)


def style_clustermap(grid, title):
    """Style a ClusterGrid to match 01_correlation_clustermap.svg:
    dendrograms on top/left, x labels at the bottom, y labels on the right."""
    grid.fig.suptitle(title, fontsize=22, fontweight='bold', y=1.02)
    ax = grid.ax_heatmap
    ax.set_xlabel('Model', fontsize=18)
    ax.set_ylabel('Model', fontsize=18)
    ax.yaxis.set_label_position('right')
    ax.tick_params(labelsize=16)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)
    # colorbar tick + label font sizes
    cax = grid.ax_cbar
    cax.tick_params(labelsize=14)
    cax.set_ylabel('p-value', fontsize=16)
    return grid


# ======================== 1. full-range clustermap ========================
OUTPUT_SVG = os.path.join(SCRIPT_DIR, '12_DeLong_Heatmap.svg')
grid = sns.clustermap(
    pval_matrix,
    row_linkage=linkage_matrix, col_linkage=linkage_matrix,
    cmap='viridis',
    vmin=0, vmax=1,
    xticklabels=model_names,
    yticklabels=model_names,
    linewidths=0.5,
    linecolor='white',
    figsize=(8, 8),
    cbar_kws={'label': 'p-value'},
)
style_clustermap(grid, 'DeLong Test')
grid.savefig(OUTPUT_SVG, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {OUTPUT_SVG}')

# ======================== 2. zoom: p in [0, 0.25] ========================
OUTPUT_ZOOM = os.path.join(SCRIPT_DIR, '12_DeLong_Heatmap_zoom.svg')
grid = sns.clustermap(
    pval_matrix,
    row_linkage=linkage_matrix, col_linkage=linkage_matrix,
    cmap='viridis',
    vmin=0, vmax=0.25,
    xticklabels=model_names,
    yticklabels=model_names,
    linewidths=0.5,
    linecolor='white',
    figsize=(8, 8),
    cbar_kws={'label': 'FDR-corrected p-value'},
)
style_clustermap(grid, 'DeLong Test')
grid.savefig(OUTPUT_ZOOM, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {OUTPUT_ZOOM}')

# ======================== 3. full matrix + significance stars ========================
OUTPUT_FULL = os.path.join(SCRIPT_DIR, '12_DeLong_Heatmap_full.svg')

annot_text = np.empty_like(pval_matrix, dtype=object)
for i in range(N):
    for j in range(N):
        p = pval_matrix[i, j]
        base = f'{p:.3f}'
        if i != j:
            if p < 0.001:
                base += ' ***'
            elif p < 0.01:
                base += ' **'
            elif p < 0.05:
                base += ' *'
        annot_text[i, j] = base

grid = sns.clustermap(
    pval_matrix,
    row_linkage=linkage_matrix, col_linkage=linkage_matrix,
    annot=annot_text, fmt='',
    cmap='viridis',
    vmin=0, vmax=0.25,
    xticklabels=model_names,
    yticklabels=model_names,
    linewidths=0.5,
    linecolor='white',
    figsize=(9, 9),
    cbar_kws={'label': 'FDR-corrected p-value'},
    annot_kws={'fontsize': 14},
)
style_clustermap(grid, 'DeLong Test')

# significance legend (viridis: darker = more significant)
legend_elements = [
    Patch(facecolor='#21918c', label='* p < 0.05'),
    Patch(facecolor='#3b528b', label='** p < 0.01'),
    Patch(facecolor='#440154', label='*** p < 0.001'),
]
grid.fig.legend(handles=legend_elements, loc='lower left',
                bbox_to_anchor=(0.02, 0.02), fontsize=14, framealpha=0.9)

grid.savefig(OUTPUT_FULL, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {OUTPUT_FULL}')

print('\nDone!')