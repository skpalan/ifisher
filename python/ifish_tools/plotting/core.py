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
    build_label_to_annotation_map,
    build_label_to_expression_map,
    find_clones_multi_dir,
    find_matching_clones,
    find_raw_mask_path,
    get_nb_cell_id,
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

# NB (neuroblast) visualization constants
NB_OUTLIER_COLOR = "#DAA520"  # Goldenrod (dark yellow)
NB_MARKER = "*"  # Star marker
NB_MARKER_SIZE = 150  # Larger than regular cells (30)
NB_EDGE_COLOR = "black"
NB_ANNOTATION_OFFSET = 20  # Z offset for text annotation


def is_nb_outlier(
    label_to_expr: dict[int, float],
    nb_id: Optional[int],
) -> tuple[bool, str]:
    """Check if NB expression is an outlier using IQR method.

    Args:
        label_to_expr: Dict mapping cell label to expression value.
        nb_id: NB cell label (or None).

    Returns:
        Tuple of (is_outlier: bool, direction: str).
        direction is 'high', 'low', or 'normal'.
    """
    if nb_id is None or nb_id not in label_to_expr:
        return False, "normal"

    nb_expr = label_to_expr[nb_id]
    other_expr = [v for k, v in label_to_expr.items() if k != nb_id]

    if len(other_expr) < 4:  # Need enough data for IQR
        return False, "normal"

    q1 = np.percentile(other_expr, 25)
    q3 = np.percentile(other_expr, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    if nb_expr > upper_bound:
        return True, "high"
    elif nb_expr < lower_bound and nb_expr > 0:  # Only flag low if non-zero
        return True, "low"

    return False, "normal"


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
    nb_id: Optional[int] = None,
    nb_is_outlier: bool = False,
    nb_direction: str = "normal",
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
        nb_id: Neuroblast cell ID (or None)
        nb_is_outlier: Whether NB is an outlier
        nb_direction: 'high', 'low', or 'normal'
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

    # If NB is outlier, plot it separately in dark yellow
    scatter2 = None
    if nb_id is not None and nb_is_outlier:
        nb_mask = labels == nb_id
        non_nb_mask = ~nb_mask

        # Plot non-NB voxels first
        if np.any(non_nb_mask):
            scatter2 = ax2.scatter(
                fg[non_nb_mask, 2],
                fg[non_nb_mask, 1],
                fg[non_nb_mask, 0],
                c=expr_values[non_nb_mask],
                cmap=cmap,
                norm=norm,
                s=1,
                alpha=0.6,
            )

        # Plot NB voxels in dark yellow
        if np.any(nb_mask):
            ax2.scatter(
                fg[nb_mask, 2],
                fg[nb_mask, 1],
                fg[nb_mask, 0],
                c=NB_OUTLIER_COLOR,
                s=2,
                alpha=0.8,
                label=f"NB ({nb_direction})",
            )
            ax2.legend(fontsize=7, loc="upper right")
    else:
        # Standard plotting without NB highlighting
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
    if scatter2 is not None:
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

        # Separate NB from other cells
        nb_idx = [i for i, c in enumerate(cell_ids) if c == nb_id]
        non_nb_idx = [i for i, c in enumerate(cell_ids) if c != nb_id]

        # Plot non-NB cells as circles
        if len(non_nb_idx) > 0:
            scatter3 = ax3.scatter(
                coords[non_nb_idx, 0],  # X
                coords[non_nb_idx, 1],  # Y
                coords[non_nb_idx, 2],  # Z
                c=expr[non_nb_idx],
                cmap=cmap,
                norm=norm,
                s=30,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
            )

        # Plot NB as star (always, but different color if outlier)
        if len(nb_idx) > 0:
            nb_expr_val = expr[nb_idx[0]]
            if nb_is_outlier:
                nb_color = NB_OUTLIER_COLOR
            else:
                # Get colormap object from string and apply normalization
                import matplotlib.cm as cm
                cmap_obj = cm.get_cmap(cmap)
                nb_color = cmap_obj(norm(nb_expr_val))

            ax3.scatter(
                coords[nb_idx, 0],  # X
                coords[nb_idx, 1],  # Y
                coords[nb_idx, 2],  # Z
                c=[nb_color],
                s=NB_MARKER_SIZE,
                marker=NB_MARKER,
                alpha=1.0,
                edgecolors=NB_EDGE_COLOR,
                linewidths=1,
                zorder=10,
            )

            # Add annotation with expression value
            nb_coord = coords[nb_idx[0]]
            annotation = f"NB: {nb_expr_val:.1f}"
            if nb_is_outlier:
                annotation += f" ({nb_direction}!)"

            ax3.text(
                nb_coord[0],
                nb_coord[1],
                nb_coord[2] + NB_ANNOTATION_OFFSET,
                annotation,
                fontsize=8,
                fontweight="bold",
                color="#8B6914",  # Dark goldenrod for text
                ha="center",
                zorder=11,
            )

        # Set aspect ratio for centroids
        fg_centroids = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0]])  # Convert to zyx
        _set_3d_equal_aspect(ax3, fg_centroids)

        # Add colorbar (use scatter3 if it exists, otherwise create from expression range)
        if len(non_nb_idx) > 0:
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
    cmap: str = "viridis",
    figsize: tuple = (20, 14),
    dpi: int = 300,
    max_voxels: int = 50000,
    nb_id: Optional[int] = None,
    nb_is_outlier: bool = False,
    nb_direction: str = "normal",
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

    # Calculate expression range for colorbar
    # If NB is outlier, exclude it from normalization to better show variation in other cells
    if nb_is_outlier and nb_id is not None and nb_id in label_to_expr:
        non_nb_expr = [v for k, v in label_to_expr.items() if k != nb_id]
        if non_nb_expr and max(non_nb_expr) > 0:
            vmax = max(non_nb_expr)
        else:
            vmax = 1
    else:
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
        nb_id=nb_id,
        nb_is_outlier=nb_is_outlier,
        nb_direction=nb_direction,
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
            nb_id=nb_id,
            nb_is_outlier=nb_is_outlier,
            nb_direction=nb_direction,
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
    cmap: str = "viridis",
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

    # Detect NB and check if it's an outlier
    nb_id = get_nb_cell_id(transform_data)
    nb_is_outlier, nb_direction = is_nb_outlier(label_to_expr, nb_id)

    if nb_is_outlier and nb_id is not None:
        nb_expr = label_to_expr.get(nb_id, 0)
        logger.info(
            f"  NB (cell {nb_id}) is {nb_direction} outlier for {gene}: {nb_expr:.1f}"
        )

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
        nb_id=nb_id,
        nb_is_outlier=nb_is_outlier,
        nb_direction=nb_direction,
    )

    return output_path


def process_clone_genes(
    adata_path: Path,
    unroll_path: Path,
    raw_mask_dir: Path,
    output_dir: Path,
    genes: Optional[list[str]] = None,
    cmap: str = "viridis",
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
    cmap: str = "viridis",
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


# ---------------------------------------------------------------------------
# Multi-clone comparison plotting (shared / per-clone color scale)
# ---------------------------------------------------------------------------

# Discrete palette for cell-type annotation panels
ANNOTATION_COLORS = {
    "AMMC": "#E41A1C",       # red
    "lPN": "#377EB8",        # blue
    "LN": "#4DAF4A",         # green
    "Ambiguous": "#FF7F00",  # orange
    "Unassigned": "#999999", # gray
}
_FALLBACK_ANNOTATION_CMAP = plt.cm.get_cmap("tab10")

# Panel-type string shortcuts → (type, render_mode) tuples
_PANEL_DEFAULTS = {
    "expression": ("expression", "voxels"),
    "centroids": ("expression", "centroids"),
    "pseudocolor": ("pseudocolor", "voxels"),
    "annotation": ("annotation", "centroids"),
}


def _clean_3d_axis(ax):
    """Remove axis labels and tick labels from a 3D axis."""
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def _get_mask_bbox_aspect(mask: np.ndarray) -> float:
    """Return height/width ratio from a 3D mask's bounding box (zyx)."""
    fg = np.argwhere(mask > 0)
    if len(fg) == 0:
        return 1.0
    z_range = max(float(fg[:, 0].max() - fg[:, 0].min()), 1.0)
    x_range = max(float(fg[:, 2].max() - fg[:, 2].min()), 1.0)
    return z_range / x_range


# ── expression panels ─────────────────────────────────────────────────

def _plot_expression_voxels(ax, mask, label_to_expr, cmap, norm,
                            max_voxels, nb_id, nb_is_outlier, nb_direction):
    """Expression-colored voxels on a 3D axis."""
    fg = np.argwhere(mask > 0)
    if len(fg) == 0:
        ax.text(0.5, 0.5, 0.5, "No data", ha="center", va="center")
        return None
    if len(fg) > max_voxels:
        rng = np.random.default_rng(42)
        fg = fg[rng.choice(len(fg), max_voxels, replace=False)]
    labels = mask[fg[:, 0], fg[:, 1], fg[:, 2]]
    expr_values = np.array([label_to_expr.get(int(lbl), 0) for lbl in labels])

    scatter = None
    if nb_id is not None and nb_is_outlier:
        nb_mask = labels == nb_id
        non_nb = ~nb_mask
        if np.any(non_nb):
            scatter = ax.scatter(fg[non_nb, 2], fg[non_nb, 1], fg[non_nb, 0],
                                 c=expr_values[non_nb], cmap=cmap, norm=norm,
                                 s=1, alpha=0.6)
        if np.any(nb_mask):
            ax.scatter(fg[nb_mask, 2], fg[nb_mask, 1], fg[nb_mask, 0],
                       c=NB_OUTLIER_COLOR, s=2, alpha=0.8,
                       label=f"NB ({nb_direction})")
            ax.legend(fontsize=7, loc="upper right")
    else:
        scatter = ax.scatter(fg[:, 2], fg[:, 1], fg[:, 0],
                             c=expr_values, cmap=cmap, norm=norm,
                             s=1, alpha=0.6)
    _set_3d_equal_aspect(ax, fg)
    _clean_3d_axis(ax)
    return scatter


def _plot_expression_centroids(ax, label_to_expr, centroids, cmap, norm,
                               nb_id, nb_is_outlier, nb_direction):
    """Expression-colored centroid scatter on a 3D axis."""
    cell_ids = [c for c in centroids if c in label_to_expr]
    if not cell_ids:
        ax.text(0.5, 0.5, 0.5, "No cells", ha="center", va="center")
        return None
    coords = np.array([centroids[c] for c in cell_ids])
    expr = np.array([label_to_expr.get(c, 0) for c in cell_ids])
    nb_idx = [i for i, c in enumerate(cell_ids) if c == nb_id]
    non_nb_idx = [i for i, c in enumerate(cell_ids) if c != nb_id]

    scatter = None
    if non_nb_idx:
        scatter = ax.scatter(coords[non_nb_idx, 0], coords[non_nb_idx, 1],
                             coords[non_nb_idx, 2],
                             c=expr[non_nb_idx], cmap=cmap, norm=norm,
                             s=30, alpha=0.8, edgecolors="white", linewidths=0.5)
    if nb_idx:
        import matplotlib.cm as mcm
        nb_val = expr[nb_idx[0]]
        nb_color = NB_OUTLIER_COLOR if nb_is_outlier else mcm.get_cmap(cmap)(norm(nb_val))
        ax.scatter(coords[nb_idx, 0], coords[nb_idx, 1], coords[nb_idx, 2],
                   c=[nb_color], s=NB_MARKER_SIZE, marker=NB_MARKER,
                   alpha=1.0, edgecolors=NB_EDGE_COLOR, linewidths=1, zorder=10)
        nc = coords[nb_idx[0]]
        txt = f"NB: {nb_val:.1f}" + (f" ({nb_direction}!)" if nb_is_outlier else "")
        ax.text(nc[0], nc[1], nc[2] + NB_ANNOTATION_OFFSET, txt,
                fontsize=8, fontweight="bold", color="#8B6914", ha="center", zorder=11)
    fg_c = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0]])
    _set_3d_equal_aspect(ax, fg_c)
    _clean_3d_axis(ax)
    return scatter


# ── pseudocolor panels ────────────────────────────────────────────────

def _plot_pseudocolor_voxels(ax, mask, max_voxels, **_kw):
    """Random-color voxels (one color per cell label)."""
    fg = np.argwhere(mask > 0)
    if len(fg) == 0:
        ax.text(0.5, 0.5, 0.5, "No data", ha="center", va="center")
        return None
    if len(fg) > max_voxels:
        rng = np.random.default_rng(42)
        fg = fg[rng.choice(len(fg), max_voxels, replace=False)]
    labels = mask[fg[:, 0], fg[:, 1], fg[:, 2]]
    n_labels = int(mask.max())
    np.random.seed(42)
    colors = np.random.rand(n_labels + 1, 3)
    colors[0] = [0.1, 0.1, 0.1]
    ax.scatter(fg[:, 2], fg[:, 1], fg[:, 0], c=colors[labels], s=1, alpha=0.5)
    _set_3d_equal_aspect(ax, fg)
    _clean_3d_axis(ax)
    return None  # no continuous colorbar


def _plot_pseudocolor_centroids(ax, centroids, label_to_expr, **_kw):
    """Random-color centroid scatter."""
    cell_ids = sorted(c for c in centroids if c in label_to_expr)
    if not cell_ids:
        ax.text(0.5, 0.5, 0.5, "No cells", ha="center", va="center")
        return None
    coords = np.array([centroids[c] for c in cell_ids])
    np.random.seed(42)
    colors = np.random.rand(max(cell_ids) + 1, 3)
    c_arr = np.array([colors[cid] for cid in cell_ids])
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=c_arr, s=30, alpha=0.8, edgecolors="white", linewidths=0.5)
    fg_c = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0]])
    _set_3d_equal_aspect(ax, fg_c)
    _clean_3d_axis(ax)
    return None


# ── annotation panels ─────────────────────────────────────────────────

def _annot_color(cat: str, idx: int) -> str:
    """Get color for an annotation category, with fallback."""
    if cat in ANNOTATION_COLORS:
        return ANNOTATION_COLORS[cat]
    rgba = _FALLBACK_ANNOTATION_CMAP(idx % 10)
    return "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


def _plot_annotation_voxels(ax, mask, label_to_annot, max_voxels, **_kw):
    """Annotation-colored voxels."""
    if not label_to_annot:
        ax.text(0.5, 0.5, 0.5, "No annotation", ha="center", va="center")
        return None
    fg = np.argwhere(mask > 0)
    if len(fg) == 0:
        ax.text(0.5, 0.5, 0.5, "No data", ha="center", va="center")
        return None
    if len(fg) > max_voxels:
        rng = np.random.default_rng(42)
        fg = fg[rng.choice(len(fg), max_voxels, replace=False)]
    labels = mask[fg[:, 0], fg[:, 1], fg[:, 2]]
    categories = sorted(set(label_to_annot.values()))
    cat_to_color = {c: _annot_color(c, i) for i, c in enumerate(categories)}
    from matplotlib.colors import to_rgba
    voxel_colors = np.array([to_rgba(cat_to_color.get(label_to_annot.get(int(l), "Unassigned"),
                                                       "#999999"))
                             for l in labels])
    ax.scatter(fg[:, 2], fg[:, 1], fg[:, 0], c=voxel_colors, s=1, alpha=0.6)
    _set_3d_equal_aspect(ax, fg)
    _clean_3d_axis(ax)
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cat_to_color[c], markersize=6, label=c)
               for c in categories]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)
    return None


def _plot_annotation_centroids(ax, centroids, label_to_annot, label_to_expr, **_kw):
    """Annotation-colored centroid scatter."""
    if not label_to_annot:
        ax.text(0.5, 0.5, 0.5, "No annotation", ha="center", va="center")
        return None
    cell_ids = [c for c in centroids if c in label_to_annot and c in label_to_expr]
    if not cell_ids:
        ax.text(0.5, 0.5, 0.5, "No cells", ha="center", va="center")
        return None
    coords = np.array([centroids[c] for c in cell_ids])
    categories = sorted(set(label_to_annot.values()))
    cat_to_color = {c: _annot_color(c, i) for i, c in enumerate(categories)}
    from matplotlib.colors import to_rgba
    colors = np.array([to_rgba(cat_to_color.get(label_to_annot.get(cid, "Unassigned"),
                                                 "#999999"))
                       for cid in cell_ids])
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=30, alpha=0.8, edgecolors="white", linewidths=0.5)
    fg_c = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0]])
    _set_3d_equal_aspect(ax, fg_c)
    _clean_3d_axis(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cat_to_color[c], markersize=6, label=c)
               for c in categories]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)
    return None


# ── normalise panels arg ──────────────────────────────────────────────

def _normalise_panels(panels):
    """Convert a user-facing panels list to [(type, mode), …]."""
    if panels is None:
        return [("expression", "voxels"), ("expression", "centroids")]
    out = []
    for item in panels:
        if isinstance(item, str):
            if item in _PANEL_DEFAULTS:
                out.append(_PANEL_DEFAULTS[item])
            else:
                raise ValueError(
                    f"Unknown panel shorthand '{item}'. "
                    f"Known: {list(_PANEL_DEFAULTS.keys())}"
                )
        else:
            out.append(tuple(item))
    return out


# ── per-clone vmax helper ─────────────────────────────────────────────

def _compute_clone_vmax(cd):
    """Compute expression vmax for a single clone dict (exclude NB outlier)."""
    expr = cd["label_to_expr"]
    nb_id = cd.get("nb_id")
    if cd.get("nb_is_outlier", False) and nb_id is not None:
        vals = [v for k, v in expr.items() if k != nb_id]
    else:
        vals = list(expr.values())
    return max(vals) if vals and max(vals) > 0 else 1.0


# ═══════════════════════════════════════════════════════════════════════
# MAIN COMPARISON FIGURE
# ═══════════════════════════════════════════════════════════════════════

def plot_multi_clone_comparison(
    clone_data: list[dict],
    gene_name: str,
    output_path: Path,
    panels: Optional[list] = None,
    color_scale: str = "shared",
    annotation_col: str = "cell_type",
    cmap: str = "viridis",
    dpi: int = 300,
    max_voxels: int = 50000,
    panel_width: Optional[float] = None,
    panel_height: Optional[float] = None,
    fontsize: Optional[float] = None,
) -> None:
    """Create a multi-clone comparison figure for one gene.

    Layout: ``len(panels)`` rows × ``len(clone_data)`` columns.
    Each ``panels`` entry becomes one row; each clone becomes one column.

    Args:
        clone_data: List of dicts, each containing:
            * ``clone_name``  (str)
            * ``unrolled_mask``  (np.ndarray, 3-D z/y/x)
            * ``label_to_expr``  (dict[int, float])
            * ``unrolled_centroids``  (dict[int, np.ndarray])
            * ``nb_id``  (Optional[int])
            * ``nb_is_outlier``  (bool)
            * ``nb_direction``  (str)
            * ``label_to_annotation``  (dict[int, str], may be empty)
        gene_name: Gene being visualised.
        output_path: Where to save the PNG.
        panels: Rows to render.  Each item is ``(panel_type, render_mode)``
            or a string shorthand (``"expression"``, ``"centroids"``,
            ``"pseudocolor"``, ``"annotation"``).
            Defaults to ``[("expression", "voxels"), ("expression",
            "centroids")]``.
        color_scale: ``"shared"`` for global vmax across clones, or
            ``"per_clone"`` for independent normalisation per clone
            (each column gets its own colorbar).
        annotation_col: Name of the annotation column (used in titles).
        cmap: Matplotlib colormap name.
        dpi: Output resolution.
        max_voxels: Maximum voxels per expression panel.
        panel_width: Width of each sub-panel in inches.  If *None* and
            ``panel_height`` is given, inferred from data aspect ratio;
            otherwise defaults to 5.
        panel_height: Height of each sub-panel in inches.  If *None* and
            ``panel_width`` is given, inferred from data aspect ratio;
            otherwise defaults to 5.
        fontsize: Base font size.  When set, all font sizes are scaled
            relative to this value.  *None* keeps the module defaults.
    """
    # ---- font size ----
    if fontsize is not None:
        rc = {
            "font.size": fontsize,
            "axes.labelsize": fontsize * 1.1,
            "axes.titlesize": fontsize * 1.2,
            "figure.titlesize": fontsize * 1.3,
            "xtick.labelsize": fontsize * 0.9,
            "ytick.labelsize": fontsize * 0.9,
        }
        plt.rcParams.update(rc)
    else:
        plt.rcParams.update(RCPARAMS)

    panels_list = _normalise_panels(panels)
    n_rows = len(panels_list)
    n_cols = len(clone_data)
    if n_cols == 0:
        logger.warning(f"No clone data for {gene_name}, skipping comparison plot")
        return

    # ---- panel size ----
    aspect = _get_mask_bbox_aspect(clone_data[0]["unrolled_mask"])
    if panel_width is not None and panel_height is not None:
        pw, ph = panel_width, panel_height
    elif panel_width is not None:
        pw = panel_width
        ph = pw * aspect
    elif panel_height is not None:
        ph = panel_height
        pw = ph / aspect if aspect > 0 else ph
    else:
        pw, ph = 5.0, 5.0

    fig_w = pw * n_cols + 1.5   # colorbar space
    fig_h = ph * n_rows + 1.0   # suptitle space

    # ---- normalisation (shared or per-clone) ----
    if color_scale == "shared":
        all_vals: list[float] = []
        for cd in clone_data:
            all_vals.extend(_v for _v in
                            ([v for k, v in cd["label_to_expr"].items()
                              if not (cd.get("nb_is_outlier") and k == cd.get("nb_id"))]
                             ))
        global_vmax = max(all_vals) if all_vals and max(all_vals) > 0 else 1.0
        norms = [Normalize(vmin=0, vmax=global_vmax)] * n_cols
        scale_label = f"shared 0–{global_vmax:.1f}"
    else:  # per_clone
        norms = []
        vmaxes = []
        for cd in clone_data:
            vm = _compute_clone_vmax(cd)
            vmaxes.append(vm)
            norms.append(Normalize(vmin=0, vmax=vm))
        scale_label = "per-clone scale"

    # ---- create figure ----
    fig = plt.figure(figsize=(fig_w, fig_h))
    title_fs = fontsize * 1.3 if fontsize else 14
    fig.suptitle(f"{gene_name}  ({scale_label})",
                 fontsize=title_fs, fontweight="bold", y=0.98)

    last_scatter_per_col: list = [None] * n_cols  # for per-clone colorbars
    last_scatter_global = None

    for row_idx, (ptype, pmode) in enumerate(panels_list):
        for col_idx, cd in enumerate(clone_data):
            ax = fig.add_subplot(n_rows, n_cols,
                                 row_idx * n_cols + col_idx + 1,
                                 projection="3d")
            norm_i = norms[col_idx]
            clone_name = cd["clone_name"]
            mask = cd["unrolled_mask"]
            l2e = cd["label_to_expr"]
            centroids = cd["unrolled_centroids"]
            nb_id = cd.get("nb_id")
            nb_out = cd.get("nb_is_outlier", False)
            nb_dir = cd.get("nb_direction", "normal")
            l2a = cd.get("label_to_annotation", {})

            sc = None
            row_label = f"{ptype} ({pmode})"

            if ptype == "expression" and pmode == "voxels":
                sc = _plot_expression_voxels(
                    ax, mask, l2e, cmap, norm_i, max_voxels,
                    nb_id, nb_out, nb_dir)
                row_label = "Expression"
            elif ptype == "expression" and pmode == "centroids":
                sc = _plot_expression_centroids(
                    ax, l2e, centroids, cmap, norm_i,
                    nb_id, nb_out, nb_dir)
                row_label = "Centroids"
            elif ptype == "pseudocolor" and pmode == "voxels":
                _plot_pseudocolor_voxels(ax, mask, max_voxels)
                row_label = "Pseudocolor"
            elif ptype == "pseudocolor" and pmode == "centroids":
                _plot_pseudocolor_centroids(ax, centroids, l2e)
                row_label = "Pseudocolor"
            elif ptype == "annotation" and pmode == "voxels":
                _plot_annotation_voxels(ax, mask, l2a, max_voxels)
                row_label = f"Annotation ({annotation_col})"
            elif ptype == "annotation" and pmode == "centroids":
                _plot_annotation_centroids(ax, centroids, l2a, l2e)
                row_label = f"Annotation ({annotation_col})"
            else:
                ax.text(0.5, 0.5, 0.5, f"Unknown\n{ptype}/{pmode}",
                        ha="center", va="center")

            if sc is not None:
                last_scatter_per_col[col_idx] = sc
                last_scatter_global = sc

            # Column title on first row only; row label appended
            title_fs_ax = fontsize * 1.1 if fontsize else 11
            if row_idx == 0:
                ax.set_title(f"{clone_name}\n{row_label}", fontsize=title_fs_ax)
            else:
                ax.set_title(row_label, fontsize=title_fs_ax)

    # ---- colorbars ----
    has_expr_row = any(pt == "expression" for pt, _ in panels_list)
    if has_expr_row:
        cbar_fs = fontsize if fontsize else 11
        if color_scale == "shared" and last_scatter_global is not None:
            cbar_ax = fig.add_axes((0.93, 0.15, 0.015, 0.65))
            cbar = fig.colorbar(last_scatter_global, cax=cbar_ax)
            cbar.set_label(f"{gene_name} expr", fontsize=cbar_fs)
        elif color_scale == "per_clone":
            for ci, sc in enumerate(last_scatter_per_col):
                if sc is None:
                    continue
                # Small colorbar below each column
                left = 0.05 + ci * (0.85 / n_cols)
                width = 0.85 / n_cols * 0.6
                cbar_ax = fig.add_axes((left + width * 0.2, 0.02, width, 0.015))
                cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
                cbar.set_label(f"0–{vmaxes[ci]:.1f}", fontsize=max(cbar_fs - 2, 6))

    # ---- save ----
    fig.subplots_adjust(left=0.01, right=0.93, top=0.94, bottom=0.04,
                        wspace=0.0, hspace=0.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved comparison: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# WORKER + ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def _process_comparison_gene_worker(
    gene: str,
    clone_infos: list[dict],
    output_dir: Path,
    cmap: str,
    dpi: int,
    max_voxels: int,
    min_match_ratio: float,
    panels: Optional[list] = None,
    color_scale: str = "shared",
    annotation_col: str = "cell_type",
    panel_width: Optional[float] = None,
    panel_height: Optional[float] = None,
    fontsize: Optional[float] = None,
) -> Optional[Path]:
    """Worker for a single gene in multi-clone comparison."""
    panels_list = _normalise_panels(panels)
    needs_annotation = any(pt == "annotation" for pt, _ in panels_list)

    clone_data_list: list[dict] = []
    for ci in clone_infos:
        clone_name: str = ci["clone_name"]
        adata = load_anndata(ci["adata_path"])

        if gene not in adata.var_names:
            logger.debug(f"Gene {gene} not in {clone_name}, skipping clone")
            continue

        try:
            label_to_expr = build_label_to_expression_map(
                adata, ci["transform_data"], gene, min_match_ratio)
        except ValueError as exc:
            logger.warning(f"Skipping {clone_name} for {gene}: {exc}")
            continue

        if not label_to_expr:
            logger.warning(f"No expression data for {gene} in {clone_name}")
            continue

        nb_id = ci["nb_id"]
        nb_is_outlier, nb_direction = is_nb_outlier(label_to_expr, nb_id)

        # Annotation map (may be empty if column absent)
        l2a: dict = {}
        if needs_annotation:
            l2a = build_label_to_annotation_map(
                adata, ci["transform_data"], annotation_col, min_match_ratio)

        clone_data_list.append({
            "clone_name": clone_name,
            "unrolled_mask": ci["unrolled_mask"],
            "label_to_expr": label_to_expr,
            "unrolled_centroids": ci["unrolled_centroids"],
            "nb_id": nb_id,
            "nb_is_outlier": nb_is_outlier,
            "nb_direction": nb_direction,
            "label_to_annotation": l2a,
        })

    if not clone_data_list:
        logger.warning(f"No valid clones for gene {gene}, skipping")
        return None

    out_path = output_dir / f"{gene}_comparison.png"
    plot_multi_clone_comparison(
        clone_data=clone_data_list,
        gene_name=gene,
        output_path=out_path,
        panels=panels,
        color_scale=color_scale,
        annotation_col=annotation_col,
        cmap=cmap,
        dpi=dpi,
        max_voxels=max_voxels,
        panel_width=panel_width,
        panel_height=panel_height,
        fontsize=fontsize,
    )
    return out_path


def process_comparison_all_genes(
    matrix_dirs: Union[str, Path, list],
    unroll_dirs: Union[str, Path, list],
    output_dir: Union[str, Path],
    clones: Optional[list[str]] = None,
    genes: Optional[list[str]] = None,
    panels: Optional[list] = None,
    color_scale: str = "shared",
    annotation_col: str = "cell_type",
    cmap: str = "viridis",
    dpi: int = 300,
    max_voxels: int = 50000,
    workers: int = 1,
    min_match_ratio: float = 0.99,
    panel_width: Optional[float] = None,
    panel_height: Optional[float] = None,
    fontsize: Optional[float] = None,
) -> list[Path]:
    """Generate multi-clone comparison plots for all (or selected) genes.

    For each gene a single PNG is produced with all clones side-by-side.

    Args:
        matrix_dirs: One or more directories containing ``.h5ad`` files
            (plain or annotated).  Accepts a single path or a list.
        unroll_dirs: One or more directories with unroll output sub-dirs.
        output_dir: Output directory for comparison PNGs.
        clones: Clone names to include (``None`` = auto-discover).
        genes: Genes to plot (``None`` = union across clones).
        panels: See :func:`plot_multi_clone_comparison`.
        color_scale: ``"shared"`` or ``"per_clone"``.
        annotation_col: obs column for annotation panels (default
            ``"cell_type"``).
        cmap: Colormap name.
        dpi: Output resolution.
        max_voxels: Max voxels per expression panel.
        workers: Parallel workers (parallelised across genes).
        min_match_ratio: Minimum cell-match ratio.
        panel_width: Per-panel width in inches (None = auto).
        panel_height: Per-panel height in inches (None = auto).
        fontsize: Base font size (None = module default).

    Returns:
        List of paths to generated PNGs.
    """
    # Normalise directory arguments
    if isinstance(matrix_dirs, (str, Path)):
        matrix_dirs = [Path(matrix_dirs)]
    else:
        matrix_dirs = [Path(d) for d in matrix_dirs]

    if isinstance(unroll_dirs, (str, Path)):
        unroll_dirs = [Path(unroll_dirs)]
    else:
        unroll_dirs = [Path(d) for d in unroll_dirs]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches = find_clones_multi_dir(matrix_dirs, unroll_dirs, clones)
    if not matches:
        logger.error("No matching clones found – nothing to plot")
        return []

    # Pre-load heavy data per clone
    clone_infos: list[dict] = []
    all_genes_set: set[str] = set()
    for clone_name, h5ad_path, unroll_path in matches:
        logger.info(f"Loading data for {clone_name} …")
        adata = load_anndata(h5ad_path)
        transform_data = load_transform_json(unroll_path / "transform.json")
        unrolled_mask = load_unrolled_mask(unroll_path / "unrolled_mask.tif")
        unrolled_centroids = get_transformed_centroids(transform_data)
        nb_id = get_nb_cell_id(transform_data)
        all_genes_set.update(adata.var_names)

        clone_infos.append({
            "clone_name": clone_name,
            "adata_path": h5ad_path,
            "transform_data": transform_data,
            "unrolled_mask": unrolled_mask,
            "unrolled_centroids": unrolled_centroids,
            "nb_id": nb_id,
        })

    gene_list = list(genes) if genes is not None else sorted(all_genes_set)
    logger.info(f"Will generate comparison plots for {len(gene_list)} gene(s)")

    # Common kwargs for the worker
    worker_kw = dict(
        cmap=cmap, dpi=dpi, max_voxels=max_voxels,
        min_match_ratio=min_match_ratio,
        panels=panels, color_scale=color_scale,
        annotation_col=annotation_col,
        panel_width=panel_width, panel_height=panel_height,
        fontsize=fontsize,
    )

    outputs: list[Path] = []
    if workers <= 1:
        for gene in tqdm(gene_list, desc="Comparison plots"):
            result = _process_comparison_gene_worker(
                gene, clone_infos, output_dir, **worker_kw)
            if result:
                outputs.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_comparison_gene_worker,
                    gene, clone_infos, output_dir, **worker_kw,
                ): gene
                for gene in gene_list
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Comparison plots"):
                gene = futures[future]
                try:
                    result = future.result()
                    if result:
                        outputs.append(result)
                except Exception as exc:
                    logger.error(f"Failed gene {gene}: {exc}")
                    raise

    logger.info(f"Comparison plots generated: {len(outputs)}")
    return outputs
