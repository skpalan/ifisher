"""I/O utilities for loading data for spatial expression plotting."""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import tifffile
from scipy.spatial import KDTree

# Regex to extract 'brainNN_cloneN' from any h5ad filename
_CLONE_RE = re.compile(r"(brain\d+_clone\d+)")

logger = logging.getLogger(__name__)


def load_anndata(path: Union[str, Path]) -> ad.AnnData:
    """Load AnnData file with count matrix.

    Args:
        path: Path to the .h5ad file.

    Returns:
        AnnData object with count matrix and metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")
    return ad.read_h5ad(path)


def load_transform_json(path: Union[str, Path]) -> dict:
    """Load transform.json with cell centroids.

    Args:
        path: Path to transform.json file.

    Returns:
        Dictionary with 'cells' mapping label -> centroid info.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Convert centroid arrays
    if "cells" in data:
        for cell_id, cell_info in data["cells"].items():
            if "centroid_transformed" in cell_info:
                cell_info["centroid_transformed"] = np.array(
                    cell_info["centroid_transformed"]
                )
            if "centroid_original" in cell_info:
                cell_info["centroid_original"] = np.array(cell_info["centroid_original"])

    return data


def load_unrolled_mask(path: Union[str, Path]) -> np.ndarray:
    """Load unrolled 3D mask from TIFF file.

    Args:
        path: Path to unrolled_mask.tif file.

    Returns:
        3D numpy array (z, y, x) with uint16 cell labels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    return tifffile.imread(path)


def get_transformed_centroids(transform_data: dict) -> dict[int, np.ndarray]:
    """Extract transformed centroids from transform.json data.

    Args:
        transform_data: Loaded transform.json data.

    Returns:
        Dict mapping cell label (int) to centroid [x, y, z] array.
    """
    centroids = {}
    cells = transform_data.get("cells", {})

    for cell_id_str, cell_info in cells.items():
        cell_id = int(cell_id_str)
        if "centroid_transformed" in cell_info:
            centroids[cell_id] = cell_info["centroid_transformed"]

    return centroids


def get_original_centroids(transform_data: dict) -> dict[int, np.ndarray]:
    """Extract original (non-transformed) centroids from transform.json data.

    Note: Transform.json stores centroid_original as [z, y, x] format.
    We convert to [x, y, z] for consistent plotting.

    Args:
        transform_data: Loaded transform.json data.

    Returns:
        Dict mapping cell label (int) to centroid [x, y, z] array.
    """
    centroids = {}
    cells = transform_data.get("cells", {})

    for cell_id_str, cell_info in cells.items():
        cell_id = int(cell_id_str)
        if "centroid_original" in cell_info:
            orig = cell_info["centroid_original"]
            # Convert [z, y, x] -> [x, y, z]
            centroids[cell_id] = np.array([orig[2], orig[1], orig[0]])

    return centroids


def validate_mask_labels(
    unrolled_mask: np.ndarray,
    raw_mask: np.ndarray,
    clone_name: str,
    min_overlap_ratio: float = 0.99,
) -> None:
    """Validate that unrolled and raw masks have matching cell labels.

    Args:
        unrolled_mask: Unrolled 3D mask (z, y, x).
        raw_mask: Raw 3D mask (z, y, x).
        clone_name: Clone name for error messages.
        min_overlap_ratio: Minimum ratio of shared labels (default: 0.99 = 99%).

    Raises:
        ValueError: If masks don't have sufficient label overlap.
    """
    unrolled_labels = set(np.unique(unrolled_mask)) - {0}  # Exclude background
    raw_labels = set(np.unique(raw_mask)) - {0}

    if len(unrolled_labels) == 0 or len(raw_labels) == 0:
        raise ValueError(f"{clone_name}: One or both masks have no cell labels")

    # Calculate overlap
    shared_labels = unrolled_labels & raw_labels
    overlap_ratio = len(shared_labels) / max(len(unrolled_labels), len(raw_labels))

    if overlap_ratio < min_overlap_ratio:
        raise ValueError(
            f"{clone_name}: Mask label mismatch - "
            f"unrolled has {len(unrolled_labels)} labels, raw has {len(raw_labels)} labels, "
            f"only {len(shared_labels)} shared ({overlap_ratio:.1%} < {min_overlap_ratio:.0%} threshold)"
        )

    logger.debug(
        f"{clone_name}: Mask validation passed - {len(shared_labels)}/{len(unrolled_labels)} "
        f"labels shared ({overlap_ratio:.1%})"
    )


def find_raw_mask_path(
    raw_mask_dir: Union[str, Path],
    clone_name: str,
) -> Optional[Path]:
    """Find raw mask file for a clone.

    Raw masks use naming pattern:
    Gel20251024_round00_brain{NN}_intact_cropped_clone{N}_useg_0202_cp_masks.tif

    Args:
        raw_mask_dir: Directory with raw mask files.
        clone_name: Clone name like 'brain08_clone1'.

    Returns:
        Path to raw mask TIFF or None if not found.
    """
    raw_mask_dir = Path(raw_mask_dir)

    # Parse clone_name: 'brain08_clone1' -> brain_id='08', clone_num='1'
    parts = clone_name.split("_")
    brain_id = parts[0].replace("brain", "")  # '08'
    clone_num = parts[1].replace("clone", "")  # '1'

    # Build expected filename pattern
    pattern = f"Gel20251024_round00_brain{brain_id}_intact_cropped_clone{clone_num}_useg_0202_cp_masks.tif"
    raw_path = raw_mask_dir / pattern

    if raw_path.exists():
        return raw_path

    logger.warning(f"Raw mask not found: {raw_path}")
    return None


def _match_cells_by_number(
    adata: ad.AnnData,
    transform_data: dict,
) -> dict[int, int]:
    """Match cells by cell number (simple matching).

    Returns:
        Dict mapping AnnData row index to transform cell label.
    """
    matches = {}
    cells_in_transform = set(int(k) for k in transform_data.get("cells", {}).keys())

    for i, obs_name in enumerate(adata.obs_names):
        try:
            cell_num = int(obs_name.split("_cell")[-1])
        except (ValueError, IndexError):
            continue
        if cell_num in cells_in_transform:
            matches[i] = cell_num

    return matches


def _match_cells_by_spatial(
    adata: ad.AnnData,
    transform_data: dict,
    tolerance: float = 5.0,
) -> dict[int, int]:
    """Match cells by spatial proximity of centroids.

    Note: AnnData stores centroids as (x, y, z).
    Transform.json stores centroid_original as [z, y, x] format.
    So: AnnData.x ~ Transform[2], AnnData.y ~ Transform[1], AnnData.z ~ Transform[0]

    Args:
        adata: AnnData with centroid_x, centroid_y, centroid_z columns.
        transform_data: Loaded transform.json data.
        tolerance: Maximum distance for a match.

    Returns:
        Dict mapping AnnData row index to transform cell label.
    """
    matches = {}
    cells = transform_data.get("cells", {})

    # Check if AnnData has centroid columns
    if not all(col in adata.obs.columns for col in ["centroid_x", "centroid_y", "centroid_z"]):
        logger.warning("AnnData missing centroid columns for spatial matching")
        return matches

    # Build array of transform centroids (convert from [z,y,x] to [x,y,z])
    transform_labels = []
    transform_coords = []
    for cell_id_str, cell_info in cells.items():
        if "centroid_original" in cell_info:
            orig = cell_info["centroid_original"]
            # Convert [z, y, x] -> [x, y, z]
            transform_labels.append(int(cell_id_str))
            transform_coords.append([orig[2], orig[1], orig[0]])

    if len(transform_coords) == 0:
        logger.warning("No centroid_original in transform.json")
        return matches

    transform_coords = np.array(transform_coords)

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(transform_coords)

    # Match each AnnData cell to nearest transform cell
    for i in range(adata.n_obs):
        adata_coord = np.array([
            adata.obs.centroid_x.iloc[i],
            adata.obs.centroid_y.iloc[i],
            adata.obs.centroid_z.iloc[i],
        ])

        dist, idx = tree.query(adata_coord)
        if dist <= tolerance:
            matches[i] = transform_labels[idx]

    return matches


def build_label_to_expression_map(
    adata: ad.AnnData,
    transform_data: dict,
    gene: str,
    min_match_ratio: float = 0.99,
) -> dict[int, float]:
    """Map cell labels to expression values for a gene.

    First tries to match cells by cell number. If that produces few matches,
    falls back to spatial matching using centroid coordinates.

    Args:
        adata: AnnData with count matrix. X[cell, gene] = count.
        transform_data: Loaded transform.json data.
        gene: Gene name to extract expression for.
        min_match_ratio: Minimum required match ratio (default: 0.99 = 99%).

    Returns:
        Dict mapping cell label (int) to expression value (float).

    Raises:
        ValueError: If cell matching fails (< min_match_ratio cells matched).
    """
    label_to_expr = {}

    # Check if gene exists
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in AnnData")

    gene_idx = list(adata.var_names).index(gene)

    # Try simple cell number matching first
    cell_matches = _match_cells_by_number(adata, transform_data)

    # If few matches, try spatial matching
    if len(cell_matches) < adata.n_obs * 0.5:
        logger.info(f"Simple matching found {len(cell_matches)}/{adata.n_obs} cells, trying spatial matching")
        spatial_matches = _match_cells_by_spatial(adata, transform_data)
        if len(spatial_matches) > len(cell_matches):
            cell_matches = spatial_matches
            logger.info(f"Spatial matching found {len(cell_matches)}/{adata.n_obs} cells")

    # Validate match ratio
    match_ratio = len(cell_matches) / adata.n_obs if adata.n_obs > 0 else 0
    if match_ratio < min_match_ratio:
        raise ValueError(
            f"Cell matching failed for {gene}: only {len(cell_matches)}/{adata.n_obs} "
            f"cells matched ({match_ratio:.1%} < {min_match_ratio:.0%} threshold). "
            f"h5ad may not match mask data."
        )

    # Build expression map
    for adata_idx, transform_label in cell_matches.items():
        expr_val = adata.X[adata_idx, gene_idx]
        if hasattr(expr_val, "toarray"):
            expr_val = expr_val.toarray().flatten()[0]
        label_to_expr[transform_label] = float(expr_val)

    return label_to_expr


def find_matching_clones(
    matrix_dir: Union[str, Path],
    unroll_dir: Union[str, Path],
) -> list[tuple[Path, Path]]:
    """Find clones that have both AnnData and unrolled mask data.

    Args:
        matrix_dir: Directory containing .h5ad files.
        unroll_dir: Directory containing brain*_clone*/ subdirectories.

    Returns:
        List of (adata_path, unroll_path) tuples for matching clones.
    """
    matrix_dir = Path(matrix_dir)
    unroll_dir = Path(unroll_dir)

    matches = []

    # Find all h5ad files
    for h5ad_path in sorted(matrix_dir.glob("*.h5ad")):
        clone_name = h5ad_path.stem  # e.g., 'brain08_clone1'
        unroll_path = unroll_dir / clone_name

        if unroll_path.exists():
            mask_path = unroll_path / "unrolled_mask.tif"
            transform_path = unroll_path / "transform.json"

            if mask_path.exists() and transform_path.exists():
                matches.append((h5ad_path, unroll_path))
                logger.info(f"Found matching clone: {clone_name}")
            else:
                logger.warning(
                    f"Clone {clone_name} directory exists but missing mask or transform"
                )
        else:
            logger.debug(f"No unroll data for {clone_name}")

    return matches


def find_clones_multi_dir(
    matrix_dirs: list[Path],
    unroll_dirs: list[Path],
    clones: Optional[list[str]] = None,
) -> list[tuple[str, Path, Path]]:
    """Find clones across multiple matrix and unroll directories.

    Searches each directory in order; first match wins for each clone name.
    If ``clones`` is provided, only those clones are returned (and a
    ``ValueError`` is raised if any cannot be found).  Otherwise all clones
    present in **both** matrix and unroll directories are returned.

    Args:
        matrix_dirs: Directories containing ``.h5ad`` files.
        unroll_dirs: Directories containing ``brain*_clone*/`` subdirectories
            (each with ``unrolled_mask.tif`` and ``transform.json``).
        clones: Optional list of clone names to include.  ``None`` means
            auto-discover (intersection of matrix and unroll directories).

    Returns:
        Sorted list of ``(clone_name, h5ad_path, unroll_path)`` tuples.

    Raises:
        ValueError: If a requested clone is missing from matrix or unroll dirs.
    """
    # Build {clone_name: h5ad_path} – first directory match wins.
    # Flexible matching: extract 'brainNN_cloneN' from any filename.
    h5ad_map: dict[str, Path] = {}
    for mdir in matrix_dirs:
        mdir = Path(mdir)
        if not mdir.exists():
            logger.warning(f"Matrix directory not found, skipping: {mdir}")
            continue
        # Track per-directory hits to detect ambiguous matches
        dir_hits: dict[str, list[Path]] = {}
        for p in sorted(mdir.glob("*.h5ad")):
            m = _CLONE_RE.search(p.stem)
            cname = m.group(1) if m else p.stem
            dir_hits.setdefault(cname, []).append(p)
        for cname, paths in dir_hits.items():
            if len(paths) > 1:
                raise ValueError(
                    f"Ambiguous h5ad match for clone '{cname}' in {mdir}: "
                    f"{[p.name for p in paths]}"
                )
            if cname not in h5ad_map:  # first-dir wins
                h5ad_map[cname] = paths[0]

    # Build {clone_name: unroll_path} – first directory match wins
    unroll_map: dict[str, Path] = {}
    for udir in unroll_dirs:
        udir = Path(udir)
        if not udir.exists():
            logger.warning(f"Unroll directory not found, skipping: {udir}")
            continue
        for subdir in sorted(udir.iterdir()):
            if not subdir.is_dir():
                continue
            mask_ok = (subdir / "unrolled_mask.tif").exists()
            xform_ok = (subdir / "transform.json").exists()
            if mask_ok and xform_ok and subdir.name not in unroll_map:
                unroll_map[subdir.name] = subdir

    # Determine which clones to use
    if clones is not None:
        missing_h5ad = [c for c in clones if c not in h5ad_map]
        missing_unroll = [c for c in clones if c not in unroll_map]
        if missing_h5ad:
            raise ValueError(
                f"h5ad not found for clone(s): {missing_h5ad}. "
                f"Searched dirs: {[str(d) for d in matrix_dirs]}"
            )
        if missing_unroll:
            raise ValueError(
                f"Unroll data not found for clone(s): {missing_unroll}. "
                f"Searched dirs: {[str(d) for d in unroll_dirs]}"
            )
        selected = clones
    else:
        selected = sorted(set(h5ad_map.keys()) & set(unroll_map.keys()))
        if not selected:
            logger.warning("No matching clones found across provided directories")

    results = [(name, h5ad_map[name], unroll_map[name]) for name in sorted(selected)]
    logger.info(
        f"Found {len(results)} clone(s) across {len(matrix_dirs)} matrix dir(s) "
        f"and {len(unroll_dirs)} unroll dir(s): {[r[0] for r in results]}"
    )
    return results


def build_label_to_annotation_map(
    adata: ad.AnnData,
    transform_data: dict,
    annotation_col: str = "cell_type",
    min_match_ratio: float = 0.99,
) -> dict[int, str]:
    """Map cell labels to annotation category strings.

    Uses the same cell-matching logic as :func:`build_label_to_expression_map`
    but reads a categorical ``obs`` column instead of expression values.

    Args:
        adata: AnnData with metadata.
        transform_data: Loaded transform.json data.
        annotation_col: Name of the ``obs`` column to use (e.g.
            ``"cell_type"``, ``"naive_cell_type"``).
        min_match_ratio: Minimum cell-match ratio.

    Returns:
        Dict mapping cell label (int) to annotation string.
        Returns **empty dict** if ``annotation_col`` is not present in
        ``adata.obs`` (graceful degradation for unannotated data).
    """
    if annotation_col not in adata.obs.columns:
        return {}

    # Reuse existing cell-matching logic
    cell_matches = _match_cells_by_number(adata, transform_data)
    if len(cell_matches) < adata.n_obs * 0.5:
        spatial_matches = _match_cells_by_spatial(adata, transform_data)
        if len(spatial_matches) > len(cell_matches):
            cell_matches = spatial_matches

    match_ratio = len(cell_matches) / adata.n_obs if adata.n_obs > 0 else 0
    if match_ratio < min_match_ratio:
        logger.warning(
            f"Cell matching for annotation too low: {len(cell_matches)}/{adata.n_obs} "
            f"({match_ratio:.1%}). Returning empty annotation map."
        )
        return {}

    label_to_annot: dict[int, str] = {}
    for adata_idx, transform_label in cell_matches.items():
        label_to_annot[transform_label] = str(adata.obs[annotation_col].iloc[adata_idx])
    return label_to_annot


def get_nb_cell_id(transform_data: dict) -> Optional[int]:
    """Get the neuroblast (NB) cell ID from transform data.

    NB is identified as the cell with ordered_anchor_pos == 0
    (at the start of the unrolled clone, representing the neuroblast).

    Args:
        transform_data: Loaded transform.json data.

    Returns:
        Cell label (int) of NB, or None if not found.
    """
    cells = transform_data.get("cells", {})
    min_pos = float("inf")
    nb_id = None

    for cell_id_str, cell_info in cells.items():
        pos = cell_info.get("ordered_anchor_pos", float("inf"))
        if pos < min_pos:
            min_pos = pos
            nb_id = int(cell_id_str)

    return nb_id
