"""QC plotting functions for count matrix analysis."""

import logging
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

# Color palette for genes
GENE_COLORS = sns.color_palette("tab20", 20)


def summarize_clone(adata: ad.AnnData, clone_name: str) -> dict:
    """Generate summary statistics for a clone.
    
    Args:
        adata: AnnData object with count matrix
        clone_name: Name of the clone (e.g., "brain08_clone1")
    
    Returns:
        Dictionary with summary statistics
    """
    total_counts = adata.X.sum(axis=1).A1
    n_genes = (adata.X > 0).sum(axis=1).A1
    
    return {
        "clone": clone_name,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "total_counts": int(total_counts.sum()),
        "median_counts_per_cell": float(np.median(total_counts)),
        "mean_counts_per_cell": float(np.mean(total_counts)),
        "median_genes_per_cell": float(np.median(n_genes)),
        "mean_genes_per_cell": float(np.mean(n_genes)),
        "zero_count_cells": int(np.sum(total_counts == 0)),
        "zero_count_genes": int(np.sum(adata.X.sum(axis=0).A1 == 0)),
    }


def plot_spatial_overlay_zoomed(
    mask_path: Path,
    puncta_paths: list[Path],
    ax,
    z_scale: int = 1,
):
    """Max-Z projection of mask focused on clone region with puncta overlaid.
    
    Args:
        mask_path: Path to clone mask TIFF
        puncta_paths: List of registered puncta CSV paths (already in cropped brain space)
        ax: Matplotlib axis
        z_scale: z-axis scaling factor (puncta_z * z_scale = mask_z)
    """
    mask = tifffile.imread(mask_path)

    # Find Z range of nonzero labels to focus projection
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        ax.text(0.5, 0.5, "Empty mask", ha="center", va="center",
                transform=ax.transAxes)
        return
    z_min, z_max = nz[0].min(), nz[0].max()
    y_min, y_max = nz[1].min(), nz[1].max()
    x_min, x_max = nz[2].min(), nz[2].max()

    # Pad by 20% of the clone extent for context
    pad_y = max(20, int((y_max - y_min) * 0.2))
    pad_x = max(20, int((x_max - x_min) * 0.2))
    y_lo = max(0, y_min - pad_y)
    y_hi = min(mask.shape[1], y_max + pad_y)
    x_lo = max(0, x_min - pad_x)
    x_hi = min(mask.shape[2], x_max + pad_x)

    # Max-Z projection of the cropped region
    sub = mask[z_min:z_max + 1, y_lo:y_hi, x_lo:x_hi]
    proj = sub.max(axis=0)

    # Display mask projection with random label colors
    n_labels = int(proj.max())
    if n_labels > 0:
        np.random.seed(42)
        colors = np.random.rand(n_labels + 1, 3)
        colors[0] = [0.1, 0.1, 0.1]  # background dark
        cmap = ListedColormap(colors)
    else:
        cmap = "gray"
    ax.imshow(proj, cmap=cmap, interpolation="nearest", origin="upper",
              extent=[x_lo, x_hi, y_hi, y_lo])

    # Overlay puncta from all genes (color-coded)
    # Puncta are already in cropped brain space, no offset needed
    gene_colors = plt.cm.tab20(np.linspace(0, 1, len(puncta_paths)))
    for i, pp in enumerate(puncta_paths):
        df = pd.read_csv(pp)
        # Extract gene name, handling _regis suffix
        m = re.search(r"gene-([^_.]+)", pp.name)
        gene = m.group(1) if m else pp.stem

        # Puncta are already in cropped brain space
        z_c = df["z"] * z_scale
        y_c = df["y"]
        x_c = df["x"]

        # Filter to clone Z range and spatial extent
        in_z = (z_c >= z_min) & (z_c <= z_max)
        in_y = (y_c >= y_lo) & (y_c <= y_hi)
        in_x = (x_c >= x_lo) & (x_c <= x_hi)
        
        # Apply filter
        sel_idx = in_z & in_y & in_x
        x_plot = x_c[sel_idx]
        y_plot = y_c[sel_idx]

        if len(x_plot) > 0:
            ax.scatter(x_plot, y_plot, s=1.5, alpha=0.4, color=gene_colors[i],
                       label=f"{gene} ({len(x_plot)})", zorder=2)

    ax.set_xlabel("X (pixels)", fontsize=9)
    ax.set_ylabel("Y (pixels)", fontsize=9)
    ax.set_title(f"Zoomed overlay (clone region, Z {z_min}–{z_max})", 
                fontsize=11, fontweight="bold")
    # Compact legend outside plot
    ax.legend(fontsize=5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              markerscale=3, framealpha=0.8)


def plot_spatial_overlay_wholebrain(
    mask_path: Path,
    puncta_paths: list[Path],
    ax,
    z_scale: int = 1,
):
    """Max-Z projection at cropped-brain scale showing clone location + puncta.
    
    Shows the full cropped brain with the clone outlined in cyan box.
    Both mask and puncta are in the same cropped brain coordinate space.
    
    Args:
        mask_path: Path to clone mask TIFF
        puncta_paths: List of registered puncta CSV paths (already in cropped brain space)
        ax: Matplotlib axis
        z_scale: z-axis scaling factor (puncta_z * z_scale = mask_z)
    """
    # Load mask to get clone extent
    mask = tifffile.imread(mask_path)
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        ax.text(0.5, 0.5, "Empty mask", ha="center", va="center",
                transform=ax.transAxes)
        return
    
    # Clone extent in cropped brain coordinates (same as mask coordinates)
    clone_row_min = nz[1].min()
    clone_row_max = nz[1].max()
    clone_col_min = nz[2].min()
    clone_col_max = nz[2].max()
    
    # Clone z-range
    clone_z_min = nz[0].min() / z_scale
    clone_z_max = nz[0].max() / z_scale
    
    # Get full cropped brain extent from mask dimensions
    brain_y_max, brain_x_max = mask.shape[1], mask.shape[2]
    brain_y_min, brain_x_min = 0, 0
    
    # Plot background (dark)
    ax.set_xlim(brain_x_min, brain_x_max)
    ax.set_ylim(brain_y_min, brain_y_max)
    ax.invert_yaxis()  # Use image convention (Y increases downward)
    ax.set_facecolor('#1a1a1a')
    
    # Draw clone bounding box (already in same coordinate space as puncta)
    rect = Rectangle((clone_col_min, clone_row_min),
                     clone_col_max - clone_col_min,
                     clone_row_max - clone_row_min,
                     linewidth=3, edgecolor='cyan', facecolor='none',
                     label='Clone region', zorder=10)
    ax.add_patch(rect)
    
    # Overlay puncta from all genes (downsample for visibility)
    # Puncta are already in cropped brain space
    gene_colors = plt.cm.tab20(np.linspace(0, 1, min(len(puncta_paths), 20)))
    for i, pp in enumerate(puncta_paths):
        df = pd.read_csv(pp)
        
        # Filter to clone z range
        in_z = (df["z"] >= clone_z_min) & (df["z"] <= clone_z_max)
        sel = df[in_z]
        
        # Downsample for performance (plot every Nth point)
        stride = max(1, len(sel) // 500)
        sel = sel.iloc[::stride]
        
        if len(sel) > 0:
            ax.scatter(sel["x"], sel["y"], s=0.5, alpha=0.3,
                      color=gene_colors[i % 20], zorder=1)
    
    ax.set_xlabel("X (pixels)", fontsize=9)
    ax.set_ylabel("Y (pixels)", fontsize=9)
    ax.set_title("Full cropped brain context (clone in cyan box)", 
                fontsize=11, fontweight="bold")
    ax.set_aspect('equal')
    
    # Add legend for clone box only
    ax.legend(fontsize=9, loc="upper right")


def plot_clone_qc(
    adata: ad.AnnData,
    clone_name: str,
    summary: dict,
    mask_path: Path,
    puncta_paths: list[Path],
    z_scale: int,
    output_path: Path,
):
    """Create comprehensive QC plot for one clone.
    
    Args:
        adata: AnnData object with count matrix
        clone_name: Name of the clone (e.g., "brain08_clone1")
        summary: Summary statistics dictionary from summarize_clone()
        mask_path: Path to clone mask TIFF
        puncta_paths: List of registered puncta CSV paths
        z_scale: z-axis scaling factor
        output_path: Path to save QC plot PNG
    """
    brain_id = adata.obs["brain_id"].iloc[0]
    clone_id = adata.obs["clone_id"].iloc[0]
    
    # Compute metrics
    total_counts = adata.X.sum(axis=1).A1
    n_genes_per_cell = (adata.X > 0).sum(axis=1).A1
    gene_expression = adata.X.toarray().T  # genes x cells
    
    # Load mask for spatial overlay
    mask = tifffile.imread(mask_path)
    
    # Create figure with GridSpec for flexible layout (3 rows)
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.4)
    
    # --- Row 0: Summary text + Total counts violin + histogram + genes per cell violin ---
    
    # Panel 1: Summary statistics (text)
    ax_summary = fig.add_subplot(gs[0, 0])
    ax_summary.axis("off")
    summary_text = f"""Clone: {clone_name}
Brain: {brain_id}, Clone: {clone_id}
━━━━━━━━━━━━━━━━━━━━━━━
Cells: {summary['n_cells']:,}
Genes: {summary['n_genes']:,}
Total counts: {summary['total_counts']:,}

Per-cell metrics:
  Median counts: {summary['median_counts_per_cell']:.1f}
  Mean counts: {summary['mean_counts_per_cell']:.1f}
  Median genes: {summary['median_genes_per_cell']:.1f}
  Mean genes: {summary['mean_genes_per_cell']:.1f}

Anomalies:
  Zero-count cells: {summary['zero_count_cells']}
  Zero-count genes: {summary['zero_count_genes']}

Coordinate space:
  Puncta: Cropped brain
  Mask: Cropped brain
  Z-scale: {z_scale}
"""
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment="top", family="monospace",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    
    # Panel 2: Total counts violin
    ax_vio1 = fig.add_subplot(gs[0, 1])
    sns.violinplot(y=total_counts, ax=ax_vio1, color="skyblue", inner="box")
    ax_vio1.set_ylabel("Total counts per cell", fontsize=10)
    ax_vio1.set_title("Counts distribution", fontsize=11, fontweight="bold")
    ax_vio1.set_xticks([])
    
    # Panel 3: Total counts histogram
    ax_hist1 = fig.add_subplot(gs[0, 2])
    ax_hist1.hist(total_counts, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    ax_hist1.set_xlabel("Total counts", fontsize=10)
    ax_hist1.set_ylabel("Frequency", fontsize=10)
    ax_hist1.set_title("Counts histogram", fontsize=11, fontweight="bold")
    ax_hist1.axvline(np.median(total_counts), color="red", linestyle="--", 
                     label=f"Median: {np.median(total_counts):.1f}")
    ax_hist1.legend(fontsize=8)
    
    # Panel 4: Genes per cell violin
    ax_vio2 = fig.add_subplot(gs[0, 3])
    sns.violinplot(y=n_genes_per_cell, ax=ax_vio2, color="lightcoral", inner="box")
    ax_vio2.set_ylabel("Genes detected per cell", fontsize=10)
    ax_vio2.set_title("Gene detection distribution", fontsize=11, fontweight="bold")
    ax_vio2.set_xticks([])
    
    # --- Row 1: Gene expression heatmap (full width) + histogram ---
    
    # Panel 5: Gene expression heatmap
    ax_heat = fig.add_subplot(gs[1, :3])
    gene_means = gene_expression.mean(axis=1)
    sorted_idx = np.argsort(gene_means)[::-1]
    gene_expression_sorted = gene_expression[sorted_idx, :]
    gene_names_sorted = adata.var_names[sorted_idx]
    
    im = ax_heat.imshow(np.log1p(gene_expression_sorted), aspect="auto", cmap="viridis", 
                       interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(gene_names_sorted)))
    ax_heat.set_yticklabels(gene_names_sorted, fontsize=8)
    ax_heat.set_xlabel("Cells", fontsize=10)
    ax_heat.set_ylabel("Genes (sorted by mean expr.)", fontsize=10)
    ax_heat.set_title("Gene expression heatmap (log1p)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, label="log1p(counts)", fraction=0.03)
    
    # Panel 6: Genes per cell histogram
    ax_hist2 = fig.add_subplot(gs[1, 3])
    ax_hist2.hist(n_genes_per_cell, bins=20, color="lightcoral", edgecolor="black", alpha=0.7)
    ax_hist2.set_xlabel("Genes detected", fontsize=10)
    ax_hist2.set_ylabel("Frequency", fontsize=10)
    ax_hist2.set_title("Gene detection histogram", fontsize=11, fontweight="bold")
    ax_hist2.axvline(np.median(n_genes_per_cell), color="red", linestyle="--",
                    label=f"Median: {np.median(n_genes_per_cell):.1f}")
    ax_hist2.legend(fontsize=8)
    
    # --- Row 2: Spatial XY zoomed + Spatial context + Full brain context + Top 10 Genes ---
    
    # Panel 7: Spatial XY zoomed (cells on mask max-proj, cropped to clone)
    ax_spatial = fig.add_subplot(gs[2, 0])
    
    # Get mask max-Z projection
    mask_maxproj = mask.max(axis=0)
    
    # Get cell coordinates
    y_coords = adata.obs["centroid_y"].values
    x_coords = adata.obs["centroid_x"].values
    
    # Get clone bounds with padding
    y_min, y_max = y_coords.min() - 20, y_coords.max() + 20
    x_min, x_max = x_coords.min() - 20, x_coords.max() + 20
    y_min, y_max = max(0, int(y_min)), min(mask_maxproj.shape[0], int(y_max))
    x_min, x_max = max(0, int(x_min)), min(mask_maxproj.shape[1], int(x_max))
    
    # Plot zoomed region (binary mask to avoid fading)
    ax_spatial.imshow(mask_maxproj[y_min:y_max, x_min:x_max] > 0, cmap="gray", alpha=0.3, origin="upper")
    scatter = ax_spatial.scatter(
        x_coords - x_min, 
        y_coords - y_min, 
        c=total_counts, 
        s=30, 
        cmap="viridis", 
        alpha=0.8, 
        edgecolors="white", 
        linewidths=0.5
    )
    ax_spatial.set_xlim(0, x_max - x_min)
    ax_spatial.set_ylim(y_max - y_min, 0)
    ax_spatial.set_xlabel("X (pixels)", fontsize=10)
    ax_spatial.set_ylabel("Y (pixels)", fontsize=10)
    ax_spatial.set_title("Spatial (XY, zoomed)", fontsize=11, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax_spatial)
    cbar.set_label("Counts", rotation=270, labelpad=15)
    
    # Panel 8: Spatial context (full brain) - full mask + cell centroids + clone bbox
    ax_context = fig.add_subplot(gs[2, 1])
    
    # Full brain mask max-projection (binary to avoid label gradient)
    mask_xy = mask.max(axis=0) > 0
    ax_context.imshow(mask_xy, cmap="gray", alpha=0.3, origin="upper")
    
    # Scatter cell centroids colored by total counts
    ax_context.scatter(
        x_coords, 
        y_coords, 
        c=total_counts, 
        s=10, 
        cmap="viridis", 
        alpha=0.8, 
        edgecolors="white", 
        linewidths=0.3
    )
    
    # Draw cyan dashed bounding box for clone region
    rect = Rectangle(
        (x_min, y_min), 
        x_max - x_min, 
        y_max - y_min,
        fill=False, 
        edgecolor="cyan", 
        linewidth=2, 
        linestyle="--"
    )
    ax_context.add_patch(rect)
    
    ax_context.set_xlim(0, mask_xy.shape[1])
    ax_context.set_ylim(mask_xy.shape[0], 0)
    ax_context.set_xlabel("X (pixels)", fontsize=10)
    ax_context.set_ylabel("Y (pixels)", fontsize=10)
    ax_context.set_title("Spatial context (full brain)", fontsize=11, fontweight="bold")
    ax_context.grid(False)
    
    # Panel 9: Full cropped brain context (with clone bounding box + puncta)
    ax_wholebrain = fig.add_subplot(gs[2, 2])
    plot_spatial_overlay_wholebrain(
        mask_path, puncta_paths, ax_wholebrain, z_scale
    )
    
    # Panel 10: Top 10 Genes table
    ax_table = fig.add_subplot(gs[2, 3])
    ax_table.axis("tight")
    ax_table.axis("off")
    
    # Compute gene statistics
    counts_per_gene = np.array(adata.X.sum(axis=0)).flatten()
    cells_per_gene = np.array((adata.X > 0).sum(axis=0)).flatten()
    gene_order = np.argsort(counts_per_gene)[::-1]
    
    top_genes = pd.DataFrame({
        "Gene": adata.var_names[gene_order[:10]],
        "Counts": counts_per_gene[gene_order[:10]].astype(int),
        "Cells": cells_per_gene[gene_order[:10]].astype(int),
        "Det%": (cells_per_gene[gene_order[:10]] / adata.n_obs * 100).astype(int)
    })
    
    table = ax_table.table(
        cellText=top_genes.values, 
        colLabels=top_genes.columns,
        cellLoc="center", 
        loc="center", 
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for i in range(len(top_genes.columns)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")
    ax_table.set_title("Top 10 Genes", fontsize=11, fontweight="bold", pad=20)
    
    # Overall title
    fig.suptitle(f"QC Report: {clone_name}", fontsize=16, fontweight="bold", y=0.995)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved QC plot: {output_path.name}")
