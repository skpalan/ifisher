"""Core logic for building gene×cell count matrices from 3D masks and puncta."""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.ndimage as ndi
import scipy.sparse as sp
import tifffile

logger = logging.getLogger(__name__)


def extract_gene_from_filename(filename: str) -> str:
    """Extract gene name from puncta CSV filename.

    Expected pattern: ..._gene-{GENE}.csv
    """
    m = re.search(r"gene-([^.]+)\.csv", filename)
    if m is None:
        raise ValueError(f"Cannot extract gene name from filename: {filename}")
    return m.group(1)


def extract_brain_from_mask(filename: str) -> str:
    """Extract brain ID from mask filename.

    Expected pattern: ...brain{NN}...
    """
    m = re.search(r"brain(\d+)", filename)
    if m is None:
        raise ValueError(f"Cannot extract brain ID from mask filename: {filename}")
    return m.group(1)


def extract_clone_from_mask(filename: str) -> str:
    """Extract clone ID from mask filename.

    Expected pattern: ...clone{N}...
    """
    m = re.search(r"clone(\d+)", filename)
    if m is None:
        raise ValueError(f"Cannot extract clone ID from mask filename: {filename}")
    return m.group(1)


def load_bbox(bbox_dir: Path, brain_id: str) -> Tuple[int, int]:
    """Load bounding box offsets from MATLAB bbox_ref.mat.

    Args:
        bbox_dir: Directory containing brain subdirectories (e.g., regis_downsamp_1121/).
        brain_id: Brain ID string (e.g., "08").

    Returns:
        Tuple of (row_offset, col_offset) as 0-based offsets for cropped coordinates.
        row_offset = bbox.xmin - 1 (MATLAB x = row = numpy dim 1)
        col_offset = bbox.ymin - 1 (MATLAB y = col = numpy dim 2)
    """
    bbox_path = bbox_dir / f"brain{brain_id}" / "ref" / "bbox_ref.mat"
    if not bbox_path.exists():
        raise FileNotFoundError(f"Bounding box file not found: {bbox_path}")

    bbox_data = sio.loadmat(bbox_path)
    bb = bbox_data["bbox"][0, 0]
    xmin = int(bb["xmin"][0, 0])  # MATLAB 1-based row min
    ymin = int(bb["ymin"][0, 0])  # MATLAB 1-based col min

    # Convert to 0-based offsets
    row_offset = xmin - 1
    col_offset = ymin - 1

    logger.debug(
        f"Brain {brain_id}: bbox offsets row={row_offset}, col={col_offset}"
    )
    return row_offset, col_offset


def detect_z_scale(mask_shape: Tuple[int, int, int], max_puncta_z: float) -> int:
    """Auto-detect z-axis scaling factor between puncta and mask coordinates.

    Round00 (mask source) has 2× z-resolution compared to other rounds (puncta).
    This function computes the scaling factor to apply to puncta z-coordinates.

    Args:
        mask_shape: Shape of the mask array (Z, Y, X).
        max_puncta_z: Maximum z-coordinate found in puncta data.

    Returns:
        Integer z-scale factor (typically 1 or 2).
    """
    # Estimate scale from ratio of mask Z dimension to puncta Z range
    # Add 1 to max_puncta_z since coordinates are 0-based
    ratio = mask_shape[0] / (max_puncta_z + 1)
    z_scale = round(ratio)

    logger.debug(
        f"Z-scale auto-detection: mask_Z={mask_shape[0]}, "
        f"max_puncta_z={max_puncta_z:.1f}, ratio={ratio:.3f} → scale={z_scale}"
    )
    return z_scale


def compute_cell_metadata(mask: np.ndarray) -> pd.DataFrame:
    """Compute cell centroids and sizes from a 3D label mask.

    Args:
        mask: 3D array (Z, Y, X) with integer cell labels. 0 = background.

    Returns:
        DataFrame with columns: label, centroid_z, centroid_y, centroid_x, cell_size.
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]

    if len(labels) == 0:
        return pd.DataFrame(
            columns=["label", "centroid_z", "centroid_y", "centroid_x", "cell_size"]
        )

    # Centroids via center_of_mass (returns list of (z, y, x) tuples)
    centroids = ndi.center_of_mass(mask > 0, mask, labels)
    centroids = np.array(centroids)  # shape (n_cells, 3)

    # Sizes via bincount
    counts = np.bincount(mask.ravel())
    sizes = counts[labels]

    df = pd.DataFrame(
        {
            "label": labels,
            "centroid_z": centroids[:, 0],
            "centroid_y": centroids[:, 1],
            "centroid_x": centroids[:, 2],
            "cell_size": sizes,
        }
    )
    return df


def assign_puncta_to_cells(
    puncta_df: pd.DataFrame,
    mask: np.ndarray,
    row_offset: int = 0,
    col_offset: int = 0,
    z_scale: int = 1,
) -> np.ndarray:
    """Assign puncta to cells by transforming coordinates and looking up mask.

    Transforms RS-FISH puncta coordinates (uncropped, round01+ z-resolution) to
    cropped mask coordinates (round00 z-resolution).

    Coordinate mapping:
        RS-FISH CSV: x=column, y=row, z=slice (ImageJ convention, 0-based)
        MATLAB bbox:  bbox.x→row(dim1), bbox.y→col(dim2), bbox.z→slice(dim0)
        Transform:    mask_row = csv_y - row_offset
                      mask_col = csv_x - col_offset
                      mask_z   = csv_z × z_scale

    Args:
        puncta_df: DataFrame with columns x, y, z (float coordinates, uncropped).
        mask: 3D label mask (Z, Y, X) in cropped coordinates.
        row_offset: 0-based row offset (bbox.xmin - 1).
        col_offset: 0-based col offset (bbox.ymin - 1).
        z_scale: Z-axis scaling factor (typically 2 for round00 masks).

    Returns:
        1D array of cell labels (0 = unassigned/background or out-of-bounds).
    """
    # Transform from uncropped to cropped coordinates
    # RS-FISH: csv_y = row, csv_x = col
    mask_row = np.round(puncta_df["y"].values - row_offset).astype(int)
    mask_col = np.round(puncta_df["x"].values - col_offset).astype(int)
    mask_z = np.round(puncta_df["z"].values * z_scale).astype(int)

    # Check bounds and create valid mask
    valid = (
        (mask_row >= 0)
        & (mask_row < mask.shape[1])
        & (mask_col >= 0)
        & (mask_col < mask.shape[2])
        & (mask_z >= 0)
        & (mask_z < mask.shape[0])
    )

    # Initialize labels array (0 = unassigned)
    labels = np.zeros(len(puncta_df), dtype=np.int32)

    # Only lookup valid puncta
    if valid.sum() > 0:
        labels[valid] = mask[
            mask_z[valid],
            mask_row[valid],
            mask_col[valid],
        ]

    return labels


def build_count_matrix(
    mask_path: Path,
    puncta_paths: list[Path],
    bbox_dir: Optional[Path] = None,
) -> Optional["anndata.AnnData"]:
    """Build a gene x cell AnnData from one clone mask and its puncta CSVs.

    Args:
        mask_path: Path to 3D mask TIFF.
        puncta_paths: List of puncta CSV paths (gene parsed from filename).
        bbox_dir: Directory containing bbox_ref.mat files (for coordinate correction).
                  If None, raw coordinates are used (no correction).

    Returns:
        AnnData with shape (n_cells, n_genes).
        obs: brain_id, clone_id, centroid_z, centroid_y, centroid_x, cell_size
        var_names: gene names
    """
    import anndata as ad

    brain_id = extract_brain_from_mask(mask_path.name)
    clone_id = extract_clone_from_mask(mask_path.name)

    logger.info(f"Loading mask: {mask_path.name}")
    mask = tifffile.imread(mask_path)

    # Cell metadata
    meta = compute_cell_metadata(mask)
    n_cells = len(meta)
    logger.info(f"  Brain {brain_id} clone {clone_id}: {n_cells} cells")

    if n_cells == 0:
        logger.warning(f"  No cells found in mask, skipping")
        return None

    # Load bbox offsets if provided
    row_offset, col_offset, z_scale = 0, 0, 1
    if bbox_dir is not None:
        row_offset, col_offset = load_bbox(bbox_dir, brain_id)

        # Detect z_scale from first puncta file
        if puncta_paths:
            sample_df = pd.read_csv(puncta_paths[0])
            max_z = sample_df["z"].max()
            z_scale = detect_z_scale(mask.shape, max_z)
            logger.info(
                f"  Coordinate correction: row_offset={row_offset}, "
                f"col_offset={col_offset}, z_scale={z_scale}"
            )

    # Map label -> row index
    label_to_idx = {int(lab): i for i, lab in enumerate(meta["label"].values)}

    # Build count dict per gene
    genes = []
    count_cols = []

    for csv_path in sorted(puncta_paths):
        gene = extract_gene_from_filename(csv_path.name)
        genes.append(gene)

        puncta_df = pd.read_csv(csv_path)
        cell_labels = assign_puncta_to_cells(
            puncta_df, mask, row_offset, col_offset, z_scale
        )

        col = np.zeros(n_cells, dtype=np.int32)
        for label in cell_labels:
            label = int(label)
            if label in label_to_idx:
                col[label_to_idx[label]] += 1

        n_unassigned = int(np.sum(cell_labels == 0))
        n_total = len(cell_labels)
        logger.info(
            f"  {gene}: {n_total} puncta, {n_unassigned} unassigned "
            f"({n_unassigned / n_total * 100:.1f}%)"
        )
        count_cols.append(col)

    # Assemble matrix (cells x genes)
    if count_cols:
        count_matrix = np.column_stack(count_cols)
    else:
        count_matrix = np.empty((n_cells, 0), dtype=np.int32)

    # Build AnnData
    cell_ids = [
        f"brain{brain_id}_clone{clone_id}_cell{int(lab)}"
        for lab in meta["label"].values
    ]

    adata = ad.AnnData(
        X=sp.csr_matrix(count_matrix),
        obs=pd.DataFrame(
            {
                "brain_id": brain_id,
                "clone_id": clone_id,
                "centroid_z": meta["centroid_z"].values,
                "centroid_y": meta["centroid_y"].values,
                "centroid_x": meta["centroid_x"].values,
                "cell_size": meta["cell_size"].values,
            },
            index=cell_ids,
        ),
        var=pd.DataFrame(index=genes),
    )

    return adata


def process_clone(
    mask_path: Path,
    puncta_dir: Path,
    output_dir: Path,
    bbox_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Process a single clone: build count matrix and save .h5ad.

    Args:
        mask_path: Path to clone mask TIFF.
        puncta_dir: Directory containing puncta CSVs.
        output_dir: Output directory for .h5ad files.
        bbox_dir: Directory containing bbox_ref.mat files (for coordinate correction).
                  If None, raw coordinates are used.

    Returns:
        Path to saved .h5ad file, or None if no cells.
    """
    brain_id = extract_brain_from_mask(mask_path.name)
    clone_id = extract_clone_from_mask(mask_path.name)

    # Find puncta CSVs for this brain
    puncta_paths = sorted(puncta_dir.glob(f"*_brain{brain_id}_*gene-*.csv"))

    if not puncta_paths:
        logger.warning(f"No puncta CSVs found for brain {brain_id}")
        return None

    logger.info(f"Found {len(puncta_paths)} puncta CSVs for brain {brain_id}")

    adata = build_count_matrix(mask_path, puncta_paths, bbox_dir)
    if adata is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"brain{brain_id}_clone{clone_id}.h5ad"
    adata.write_h5ad(out_path)
    logger.info(f"Saved: {out_path} ({adata.n_obs} cells x {adata.n_vars} genes)")

    return out_path
