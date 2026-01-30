"""Core logic for clone mask extraction with morphological closing."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_closing, distance_transform_edt
from skimage.morphology import ball
from tqdm import tqdm

from .config import BoxCoords, BrainCloneSpec, CloneExtractConfig, load_bbox_from_mat


def detect_mask_space(mask: np.ndarray, bbox: BoxCoords) -> str:
    """Detect whether mask is whole-brain or B-box sized.

    Parameters
    ----------
    mask : np.ndarray
        3D segmentation mask (Z, Y, X).
    bbox : BoxCoords
        B-box in whole-brain coordinates.

    Returns
    -------
    str
        "whole_brain" or "bbox"

    Raises
    ------
    ValueError
        If mask dimensions don't match either space.
    """
    mask_shape = mask.shape
    bbox_shape = bbox.shape()

    if mask_shape == bbox_shape:
        return "bbox"

    if (mask_shape[0] >= bbox.zmax and
        mask_shape[1] >= bbox.ymax and
        mask_shape[2] >= bbox.xmax):
        return "whole_brain"

    raise ValueError(
        f"Mask shape {mask_shape} doesn't match B-box shape {bbox_shape} "
        f"and is not large enough for whole-brain (need >= "
        f"({bbox.zmax}, {bbox.ymax}, {bbox.xmax})). "
        f"Cannot determine coordinate space."
    )


def compute_relative_cbox(cbox: BoxCoords, bbox: BoxCoords) -> BoxCoords:
    """Convert C-box from whole-brain to B-box-relative coordinates.

    Parameters
    ----------
    cbox : BoxCoords
        Clone bounding box in whole-brain coordinates.
    bbox : BoxCoords
        Cropping bounding box in whole-brain coordinates.

    Returns
    -------
    BoxCoords
        C-box relative to B-box origin.

    Raises
    ------
    ValueError
        If C-box extends outside B-box.
    """
    rel = BoxCoords(
        ymin=cbox.ymin - bbox.ymin,
        ymax=cbox.ymax - bbox.ymin,
        xmin=cbox.xmin - bbox.xmin,
        xmax=cbox.xmax - bbox.xmin,
        zmin=cbox.zmin - bbox.zmin,
        zmax=cbox.zmax - bbox.zmin,
    )
    bbox_shape = bbox.shape()  # (Z, Y, X)
    if (rel.ymin < 0 or rel.xmin < 0 or rel.zmin < 0 or
        rel.ymax > bbox_shape[1] or rel.xmax > bbox_shape[2] or
        rel.zmax > bbox_shape[0]):
        raise ValueError(
            f"C-box extends outside B-box. "
            f"Relative coords: y=[{rel.ymin},{rel.ymax}], "
            f"x=[{rel.xmin},{rel.xmax}], z=[{rel.zmin},{rel.zmax}], "
            f"B-box shape (Z,Y,X): {bbox_shape}"
        )
    return rel


def _resolve_overlaps(new_claims: dict, raw_seg: np.ndarray) -> np.ndarray:
    """Resolve overlapping closure claims via distance transform."""
    result = raw_seg.copy()

    total_claims = np.zeros_like(raw_seg, dtype=np.int32)
    for mask in new_claims.values():
        total_claims += mask.astype(np.int32)

    disputed = total_claims > 1

    if not disputed.any():
        for label, mask in new_claims.items():
            result[mask] = label
        return result

    distance_maps = {}
    for label in new_claims:
        distance_maps[label] = distance_transform_edt(~(raw_seg == label))

    disputed_coords = np.argwhere(disputed)
    for coord in disputed_coords:
        z, y, x = coord
        min_dist = np.inf
        winner = 0
        for label, mask in new_claims.items():
            if not mask[z, y, x]:
                continue
            dist = distance_maps[label][z, y, x]
            if dist < min_dist:
                min_dist = dist
                winner = label
        result[z, y, x] = winner

    undisputed = total_claims == 1
    for label, mask in new_claims.items():
        result[mask & undisputed] = label

    return result


def apply_morphological_closing(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Apply per-label morphological closing with overlap resolution.

    Parameters
    ----------
    mask : np.ndarray
        3D segmentation mask (Z, Y, X), integer labels.
    radius : int
        Ball radius for structuring element. 0 = no-op.

    Returns
    -------
    np.ndarray
        Closed segmentation mask.
    """
    if radius <= 0:
        return mask.copy()

    struct = ball(radius)
    labels = np.unique(mask)
    labels = labels[labels > 0]

    if len(labels) == 0:
        return mask.copy()

    new_claims = {}
    for label in tqdm(labels, desc=f"Closing (r={radius})", leave=False):
        binary_mask = mask == label
        closed = binary_closing(binary_mask, structure=struct)
        newly_claimed = closed & (mask == 0)
        if newly_claimed.any():
            new_claims[label] = newly_claimed

    if not new_claims:
        return mask.copy()

    return _resolve_overlaps(new_claims, mask)


def extract_clone_mask(
    mask: np.ndarray,
    cbox: BoxCoords,
    bbox: BoxCoords,
    mask_space: str,
    closing_radius: int = 5,
) -> np.ndarray:
    """Extract a single clone mask in B-box dimensions.

    Parameters
    ----------
    mask : np.ndarray
        Input segmentation mask (whole-brain or B-box sized).
    cbox : BoxCoords
        Clone bounding box in whole-brain coordinates.
    bbox : BoxCoords
        Cropping bounding box in whole-brain coordinates.
    mask_space : str
        "whole_brain" or "bbox".
    closing_radius : int
        Morphological closing radius.

    Returns
    -------
    np.ndarray
        B-box-sized mask with clone region populated.
    """
    if mask_space == "whole_brain":
        clone_crop = mask[cbox.to_slices()].copy()
        rel_cbox = compute_relative_cbox(cbox, bbox)
    else:
        rel_cbox = compute_relative_cbox(cbox, bbox)
        clone_crop = mask[rel_cbox.to_slices()].copy()

    if closing_radius > 0:
        clone_crop = apply_morphological_closing(clone_crop, radius=closing_radius)

    output = np.zeros(bbox.shape(), dtype=mask.dtype)
    output[rel_cbox.to_slices()] = clone_crop

    return output


def process_brain(
    spec: BrainCloneSpec,
    output_dir: str,
    closing_radius: int = 5,
    date_tag: str = "0129",
    naming_template: str = "{brain_base}_{clone_name}_useg_{date}_cp_masks.tif",
) -> list[str]:
    """Process all clones for a single brain.

    Returns list of output file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = Path(spec.mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {spec.mask_path}")

    print(f"Loading mask: {spec.mask_path}")
    mask = tifffile.imread(spec.mask_path)
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")

    mask_space = detect_mask_space(mask, spec.bbox)
    print(f"  Detected mask space: {mask_space}")

    stem = Path(spec.mask_path).stem
    brain_base = stem[:-len("_cp_masks")] if stem.endswith("_cp_masks") else stem

    output_paths = []
    for clone_name, cbox in spec.clones.items():
        print(f"\n  Processing {clone_name}...")
        t0 = time.time()

        result = extract_clone_mask(
            mask, cbox, spec.bbox,
            mask_space=mask_space,
            closing_radius=closing_radius,
        )

        n_labels = len(np.unique(result)) - 1
        n_voxels = int(np.sum(result > 0))
        elapsed = time.time() - t0
        print(f"    Cells: {n_labels}, Labeled voxels: {n_voxels}, Time: {elapsed:.1f}s")

        filename = naming_template.format(
            brain_base=brain_base,
            clone_name=clone_name,
            date=date_tag,
        )
        out_path = str(out_dir / filename)
        tifffile.imwrite(out_path, result.astype(np.uint16), compression="zlib")
        print(f"    Saved: {filename}")
        output_paths.append(out_path)

    return output_paths


def run_pipeline(config: CloneExtractConfig) -> list[str]:
    """Run the full clone extraction pipeline."""
    all_paths = []
    for spec in config.brains:
        paths = process_brain(
            spec,
            output_dir=config.output_dir,
            closing_radius=config.closing_radius,
            date_tag=config.date_tag,
            naming_template=config.naming_template,
        )
        all_paths.extend(paths)
    return all_paths
