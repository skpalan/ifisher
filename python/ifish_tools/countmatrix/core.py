"""Core logic for building gene×cell count matrices from 3D masks and puncta."""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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
    puncta_df: pd.DataFrame, mask: np.ndarray
) -> np.ndarray:
    """Assign puncta to cells by looking up mask at each punctum's coordinates.

    Args:
        puncta_df: DataFrame with columns x, y, z (float coordinates).
        mask: 3D label mask (Z, Y, X).

    Returns:
        1D array of cell labels (0 = unassigned/background).
    """
    z = np.round(puncta_df["z"].values).astype(int)
    y = np.round(puncta_df["y"].values).astype(int)
    x = np.round(puncta_df["x"].values).astype(int)

    # Clip to mask bounds
    z = np.clip(z, 0, mask.shape[0] - 1)
    y = np.clip(y, 0, mask.shape[1] - 1)
    x = np.clip(x, 0, mask.shape[2] - 1)

    return mask[z, y, x]


def build_count_matrix(
    mask_path: Path,
    puncta_paths: list[Path],
) -> Optional["anndata.AnnData"]:
    """Build a gene x cell AnnData from one clone mask and its puncta CSVs.

    Args:
        mask_path: Path to 3D mask TIFF.
        puncta_paths: List of puncta CSV paths (gene parsed from filename).

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

    # Map label -> row index
    label_to_idx = {int(lab): i for i, lab in enumerate(meta["label"].values)}

    # Build count dict per gene
    genes = []
    count_cols = []

    for csv_path in sorted(puncta_paths):
        gene = extract_gene_from_filename(csv_path.name)
        genes.append(gene)

        puncta_df = pd.read_csv(csv_path)
        cell_labels = assign_puncta_to_cells(puncta_df, mask)

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
) -> Optional[Path]:
    """Process a single clone: build count matrix and save .h5ad.

    Args:
        mask_path: Path to clone mask TIFF.
        puncta_dir: Directory containing puncta CSVs.
        output_dir: Output directory for .h5ad files.

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

    adata = build_count_matrix(mask_path, puncta_paths)
    if adata is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"brain{brain_id}_clone{clone_id}.h5ad"
    adata.write_h5ad(out_path)
    logger.info(f"Saved: {out_path} ({adata.n_obs} cells x {adata.n_vars} genes)")

    return out_path
