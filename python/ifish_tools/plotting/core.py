"""Core 3D plotting functions for spatial gene expression visualization."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server use

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import Normalize
from tqdm import tqdm

from .io import (
    build_label_to_expression_map,
    find_matching_clones,
    find_raw_mask_path,
    get_original_centroids,
    get_transformed_centroids,
    load_anndata,
    load_transform_json,
    load_unrolled_mask,
    validate_mask_labels,
)

logger = logging.getLogger(__name__)

# Publication-quality settings
RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def _set_3d_equal_aspect(ax, fg_data):
    """Set equal aspect ratio for 3D plot based on data ranges.

    Args:
        ax: 3D matplotlib axis
        fg_data: Voxel coordinates (N, 3) in [Z, Y, X] order
    """
    if len(fg_data) == 0:
        return

    z_range = float(fg_data[:, 0].max() - fg_data[:, 0].min())
    y_range = float(fg_data[:, 1].max() - fg_data[:, 1].min())
    x_range = float(fg_data[:, 2].max() - fg_data[:, 2].min())

    # Ensure no zero ranges
    x_range = max(x_range, 1.0)
    y_range = max(y_range, 1.0)
    z_range = max(z_range, 1.0)

    ax.set_box_aspect([x_range, y_range, z_range])


def _plot_mask_row(
    axes: list,
    mask: np.ndarray,
    label_to_expr: dict[int, float],
    centroids: dict[int, np.ndarray],
    row_title: str,
    gene_name: str,
    cmap: str,
    norm: Normalize,
    max_voxels: int,
):
    """Plot a row of 3 panels for a mask (random colors, expression, centroids).

    Args:
        axes: List of 3 matplotlib 3D axes [ax1, ax2, ax3]
        mask: 3D numpy array (z, y, x) with cell labels
        label_to_expr: Dict mapping cell label to expression value
        centroids: Dict mapping cell label to [x, y, z] coordinates
        row_title: Title prefix for this row (e.g., "Unrolled" or "Raw")
        gene_name: Name of the gene
        cmap: Colormap for expression
        norm: Normalization for expression values
        max_voxels: Maximum voxels to plot
    """
    ax1, ax2, ax3 = axes

    # Get foreground voxel coordinates
    fg = np.argwhere(mask > 0)  # (N, 3) in zyx order

    if len(fg) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 0.5, "No data", ha="center", va="center")
        return

    # Subsample if too many voxels
    if len(fg) > max_voxels:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(fg), max_voxels, replace=False)
        fg = fg[idx]

    # Get cell labels at sampled positions
    labels = mask[fg[:, 0], fg[:, 1], fg[:, 2]]

    # ========================
    # Panel 1: Random Colors
    # ========================
    n_labels = int(mask.max())
    np.random.seed(42)
    random_colors = np.random.rand(n_labels + 1, 3)
    random_colors[0] = [0.1, 0.1, 0.1]  # Background dark
    voxel_colors = random_colors[labels]

    ax1.scatter(
        fg[:, 2],  # X
        fg[:, 1],  # Y
        fg[:, 0],  # Z
        c=voxel_colors,
        s=1,
        alpha=0.5,
    )
    _set_3d_equal_aspect(ax1, fg)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"{row_title} Masks")

    # ========================
    # Panel 2: Expression Colors
    # ========================
    expr_values = np.zeros(len(labels), dtype=float)
    for i, lbl in enumerate(labels):
        expr_values[i] = label_to_expr.get(lbl, 0)

    scatter2 = ax2.scatter(
        fg[:, 2],
        fg[:, 1],
        fg[:, 0],
        c=expr_values,
        cmap=cmap,
        norm=norm,
        s=1,
        alpha=0.6,
    )
    _set_3d_equal_aspect(ax2, fg)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.1)
    cbar2.set_label("Expr", fontsize=9)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title(f"{row_title} - {gene_name}")

    # ========================
    # Panel 3: Centroids Only
    # ========================
    cell_ids = [c for c in centroids.keys() if c in label_to_expr]
    if len(cell_ids) > 0:
        coords = np.array([centroids[c] for c in cell_ids])  # (N, 3) xyz
        expr = np.array([label_to_expr.get(c, 0) for c in cell_ids])

        scatter3 = ax3.scatter(
            coords[:, 0],  # X
            coords[:, 1],  # Y
            coords[:, 2],  # Z
            c=expr,
            cmap=cmap,
            norm=norm,
            s=30,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )
        # Set aspect ratio for centroids
        fg_centroids = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0]])  # Convert to zyx
        _set_3d_equal_aspect(ax3, fg_centroids)
        cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.5, pad=0.1)
        cbar3.set_label("Expr", fontsize=9)
    else:
        ax3.text(0.5, 0.5, 0.5, "No cells", ha="center", va="center")

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title(f"{row_title} Centroids")


def plot_spatial_expression_3d(
    unrolled_mask: np.ndarray,
    raw_mask: Optional[np.ndarray],
    label_to_expr: dict[int, float],
    unrolled_centroids: dict[int, np.ndarray],
    original_centroids: dict[int, np.ndarray],
    gene_name: str,
    clone_name: str,
    output_path: Path,
    cmap: str = "RdYlGn",
    figsize: tuple = (20, 14),
    dpi: int = 300,
    max_voxels: int = 50000,
) -> None:
    """Create 2×3 panel 3D visualization of gene expression.

    Row 1 (Unrolled): Unrolled masks with random colors, expression, centroids
    Row 2 (Raw): Raw tissue masks with random colors, expression, centroids

    Args:
        unrolled_mask: Unrolled 3D mask (z, y, x) with cell labels.
        raw_mask: Raw (non-unrolled) 3D mask or None.
        label_to_expr: Dict mapping cell label to expression value.
        unrolled_centroids: Dict mapping cell label to transformed [x, y, z] coords.
        original_centroids: Dict mapping cell label to original [x, y, z] coords.
        gene_name: Name of the gene being visualized.
        clone_name: Name of the clone (e.g., 'brain08_clone1').
        output_path: Path to save the output PNG.
        cmap: Colormap for expression values.
        figsize: Figure size (width, height) in inches.
        dpi: Output resolution.
        max_voxels: Maximum voxels to plot per panel (subsampled for performance).
    """
    plt.rcParams.update(RCPARAMS)

    # Calculate global expression range for consistent colorbar
    all_expr = list(label_to_expr.values())
    vmax = max(all_expr) if all_expr and max(all_expr) > 0 else 1
    norm = Normalize(vmin=0, vmax=vmax)

    # Create figure with 2×3 subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{clone_name} - {gene_name}", fontsize=14, fontweight="bold")

    # Row 1: Unrolled masks
    ax1 = fig.add_subplot(231, projection="3d")
    ax2 = fig.add_subplot(232, projection="3d")
    ax3 = fig.add_subplot(233, projection="3d")

    _plot_mask_row(
        axes=[ax1, ax2, ax3],
        mask=unrolled_mask,
        label_to_expr=label_to_expr,
        centroids=unrolled_centroids,
        row_title="Unrolled",
        gene_name=gene_name,
        cmap=cmap,
        norm=norm,
        max_voxels=max_voxels,
    )

    # Row 2: Raw masks
    ax4 = fig.add_subplot(234, projection="3d")
    ax5 = fig.add_subplot(235, projection="3d")
    ax6 = fig.add_subplot(236, projection="3d")

    if raw_mask is not None:
        _plot_mask_row(
            axes=[ax4, ax5, ax6],
            mask=raw_mask,
            label_to_expr=label_to_expr,
            centroids=original_centroids,
            row_title="Raw",
            gene_name=gene_name,
            cmap=cmap,
            norm=norm,
            max_voxels=max_voxels,
        )
    else:
        for ax in [ax4, ax5, ax6]:
            ax.text(0.5, 0.5, 0.5, "Raw mask\nnot available", ha="center", va="center")
            ax.set_title("Raw (N/A)")

    # Adjust layout and save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved: {output_path}")


def process_clone_gene(
    adata_path: Path,
    unroll_path: Path,
    raw_mask_dir: Path,
    output_dir: Path,
    gene: str,
    cmap: str = "RdYlGn",
    dpi: int = 300,
    max_voxels: int = 50000,
    min_match_ratio: float = 0.99,
    min_mask_overlap: float = 0.99,
) -> Optional[Path]:
    """Process a single gene for a single clone.

    Args:
        adata_path: Path to .h5ad file.
        unroll_path: Path to unroll directory (contains unrolled_mask.tif, transform.json).
        raw_mask_dir: Directory containing raw mask files.
        output_dir: Base output directory.
        gene: Gene name to process.
        cmap: Colormap for expression.
        dpi: Output DPI.
        max_voxels: Max voxels to plot.
        min_match_ratio: Minimum cell match ratio for h5ad↔transform.json.
        min_mask_overlap: Minimum label overlap ratio for unrolled↔raw masks.

    Returns:
        Path to output file if successful, None otherwise.

    Raises:
        ValueError: If validation fails (cell matching or mask overlap below threshold).
    """
    clone_name = adata_path.stem

    # Load data
    adata = load_anndata(adata_path)
    transform_data = load_transform_json(unroll_path / "transform.json")
    unrolled_mask = load_unrolled_mask(unroll_path / "unrolled_mask.tif")

    # Load raw mask
    raw_mask_path = find_raw_mask_path(raw_mask_dir, clone_name)
    raw_mask = tifffile.imread(raw_mask_path) if raw_mask_path else None

    # Validate mask label matching
    if raw_mask is not None:
        validate_mask_labels(unrolled_mask, raw_mask, clone_name, min_mask_overlap)

    # Build expression map (will raise ValueError if matching fails)
    label_to_expr = build_label_to_expression_map(
        adata, transform_data, gene, min_match_ratio
    )

    # Get both types of centroids
    unrolled_centroids = get_transformed_centroids(transform_data)
    original_centroids = get_original_centroids(transform_data)

    if len(label_to_expr) == 0:
        raise ValueError(f"No expression data for {gene} in {clone_name}")

    # Create output path: output_dir/gene/clone_gene.png
    gene_dir = output_dir / gene
    output_path = gene_dir / f"{clone_name}_{gene}.png"

    # Generate plot
    plot_spatial_expression_3d(
        unrolled_mask=unrolled_mask,
        raw_mask=raw_mask,
        label_to_expr=label_to_expr,
        unrolled_centroids=unrolled_centroids,
        original_centroids=original_centroids,
        gene_name=gene,
        clone_name=clone_name,
        output_path=output_path,
        cmap=cmap,
        dpi=dpi,
        max_voxels=max_voxels,
    )

    return output_path


def process_clone_genes(
    adata_path: Path,
    unroll_path: Path,
    raw_mask_dir: Path,
    output_dir: Path,
    genes: Optional[list[str]] = None,
    cmap: str = "RdYlGn",
    dpi: int = 300,
    max_voxels: int = 50000,
    min_match_ratio: float = 0.99,
    min_mask_overlap: float = 0.99,
) -> list[Path]:
    """Process all genes for a single clone.

    Args:
        adata_path: Path to .h5ad file.
        unroll_path: Path to unroll directory.
        raw_mask_dir: Directory containing raw mask files.
        output_dir: Base output directory.
        genes: List of genes to process (None = all genes in AnnData).
        cmap: Colormap for expression.
        dpi: Output DPI.
        max_voxels: Max voxels to plot.
        min_match_ratio: Minimum cell match ratio for h5ad↔transform.json.
        min_mask_overlap: Minimum label overlap ratio for unrolled↔raw masks.

    Returns:
        List of output paths.

    Raises:
        ValueError: If validation fails for first gene (stops processing clone).
    """
    clone_name = adata_path.stem
    logger.info(f"Processing clone: {clone_name}")

    # Load AnnData to get gene list
    adata = load_anndata(adata_path)

    if genes is None:
        genes = list(adata.var_names)

    outputs = []
    for gene in tqdm(genes, desc=clone_name):
        result = process_clone_gene(
            adata_path=adata_path,
            unroll_path=unroll_path,
            raw_mask_dir=raw_mask_dir,
            output_dir=output_dir,
            gene=gene,
            cmap=cmap,
            dpi=dpi,
            max_voxels=max_voxels,
            min_match_ratio=min_match_ratio,
            min_mask_overlap=min_mask_overlap,
        )
        if result:
            outputs.append(result)

    return outputs


def process_all_clones(
    matrix_dir: Union[str, Path],
    unroll_dir: Union[str, Path],
    raw_mask_dir: Union[str, Path],
    output_dir: Union[str, Path],
    genes: Optional[list[str]] = None,
    cmap: str = "RdYlGn",
    dpi: int = 300,
    max_voxels: int = 50000,
    workers: int = 1,
    min_match_ratio: float = 0.99,
    min_mask_overlap: float = 0.99,
) -> list[Path]:
    """Process all matching clones.

    Args:
        matrix_dir: Directory with .h5ad files.
        unroll_dir: Directory with unroll output subdirectories.
        raw_mask_dir: Directory containing raw mask files.
        output_dir: Output directory for plots.
        genes: List of genes to process (None = all genes).
        cmap: Colormap for expression.
        dpi: Output DPI.
        max_voxels: Max voxels to plot.
        workers: Number of parallel workers.
        min_match_ratio: Minimum cell match ratio for h5ad↔transform.json.
        min_mask_overlap: Minimum label overlap ratio for unrolled↔raw masks.

    Returns:
        List of all output paths.

    Raises:
        ValueError: If any clone fails validation.
    """
    matrix_dir = Path(matrix_dir)
    unroll_dir = Path(unroll_dir)
    raw_mask_dir = Path(raw_mask_dir)
    output_dir = Path(output_dir)

    # Find matching clones
    matches = find_matching_clones(matrix_dir, unroll_dir)

    if len(matches) == 0:
        logger.error("No matching clones found!")
        return []

    logger.info(f"Found {len(matches)} matching clones")

    all_outputs = []

    if workers == 1:
        # Sequential processing
        for adata_path, unroll_path in matches:
            outputs = process_clone_genes(
                adata_path=adata_path,
                unroll_path=unroll_path,
                raw_mask_dir=raw_mask_dir,
                output_dir=output_dir,
                genes=genes,
                cmap=cmap,
                dpi=dpi,
                max_voxels=max_voxels,
                min_match_ratio=min_match_ratio,
                min_mask_overlap=min_mask_overlap,
            )
            all_outputs.extend(outputs)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for adata_path, unroll_path in matches:
                future = executor.submit(
                    process_clone_genes,
                    adata_path=adata_path,
                    unroll_path=unroll_path,
                    raw_mask_dir=raw_mask_dir,
                    output_dir=output_dir,
                    genes=genes,
                    cmap=cmap,
                    dpi=dpi,
                    max_voxels=max_voxels,
                    min_match_ratio=min_match_ratio,
                    min_mask_overlap=min_mask_overlap,
                )
                futures[future] = adata_path.stem

            for future in as_completed(futures):
                clone_name = futures[future]
                try:
                    outputs = future.result()
                    all_outputs.extend(outputs)
                    logger.info(f"Completed {clone_name}: {len(outputs)} plots")
                except Exception as e:
                    logger.error(f"Failed {clone_name}: {e}")
                    raise  # Re-raise to stop processing

    logger.info(f"Total plots generated: {len(all_outputs)}")
    return all_outputs
