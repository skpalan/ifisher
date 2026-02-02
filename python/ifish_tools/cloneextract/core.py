"""Core logic for clone mask extraction with morphological closing."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_closing, distance_transform_edt
from skimage.morphology import ball
from tqdm import tqdm

from .config import BoxCoords, BrainCloneSpec, CloneExtractConfig, load_bbox_from_mat

try:
    import cupy as cp
    from cupyx.scipy.ndimage import binary_closing as gpu_binary_closing
    from cupyx.scipy.ndimage import distance_transform_edt as gpu_distance_transform_edt
    HAS_GPU = True
except (ImportError, Exception):
    HAS_GPU = False


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

    # Exact match in all dimensions
    if mask_shape == bbox_shape:
        return "bbox"

    # Check if Y and X dimensions match bbox (hybrid case: bbox-cropped in Y/X, different Z resolution)
    if mask_shape[1] == bbox_shape[1] and mask_shape[2] == bbox_shape[2]:
        return "bbox"

    # Whole-brain: mask is larger than bbox in all dimensions
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
    """Resolve overlapping closure claims via distance transform (CPU)."""
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


def _resolve_overlaps_gpu(new_claims: dict, raw_seg) -> np.ndarray:
    """Resolve overlapping closure claims via distance transform (GPU).

    All inputs/outputs are CuPy arrays on GPU. Returns a NumPy array.
    Uses a streaming min-distance approach to avoid allocating a full
    (N_labels x Z x Y x X) stack that can exceed GPU memory.
    """
    result = raw_seg.copy()
    labels = list(new_claims.keys())

    total_claims = cp.zeros_like(raw_seg, dtype=cp.int32)
    for mask in new_claims.values():
        total_claims += mask.astype(cp.int32)

    disputed = total_claims > 1

    if not disputed.any():
        for label, mask in new_claims.items():
            result[mask] = label
        return cp.asnumpy(result)

    # Assign undisputed claims first
    undisputed = total_claims == 1
    for label, mask in new_claims.items():
        result[mask & undisputed] = label

    # For disputed voxels: stream through labels, track running min distance
    best_dist = cp.full(raw_seg.shape, cp.inf, dtype=cp.float32)
    best_label = cp.zeros(raw_seg.shape, dtype=raw_seg.dtype)

    for label in tqdm(labels, desc="Distance transforms (GPU)", leave=False):
        dt = gpu_distance_transform_edt(~(raw_seg == label)).astype(cp.float32)
        claim = new_claims[label]
        # Only consider where this label actually claims AND is disputed
        update_mask = claim & disputed & (dt < best_dist)
        best_dist[update_mask] = dt[update_mask]
        best_label[update_mask] = label
        del dt

    result[disputed] = best_label[disputed]
    del best_dist, best_label

    return cp.asnumpy(result)


def apply_morphological_closing_gpu(
    mask: np.ndarray, radius: int = 5, device: int = 0,
) -> np.ndarray:
    """GPU-accelerated per-label morphological closing with overlap resolution.

    Parameters
    ----------
    mask : np.ndarray
        3D segmentation mask (Z, Y, X), integer labels (NumPy on CPU).
    radius : int
        Ball radius for structuring element. 0 = no-op.
    device : int
        CUDA device index.

    Returns
    -------
    np.ndarray
        Closed segmentation mask (NumPy on CPU).
    """
    if radius <= 0:
        return mask.copy()

    labels_np = np.unique(mask)
    labels_np = labels_np[labels_np > 0]
    if len(labels_np) == 0:
        return mask.copy()

    with cp.cuda.Device(device):
        struct_gpu = cp.array(ball(radius))
        mask_gpu = cp.array(mask)
        background = mask_gpu == 0

        new_claims = {}
        for label in tqdm(labels_np, desc=f"Closing (r={radius}, GPU)", leave=False):
            binary_mask = mask_gpu == label
            closed = gpu_binary_closing(binary_mask, structure=struct_gpu)
            newly_claimed = closed & background
            if newly_claimed.any():
                new_claims[int(label)] = newly_claimed

        if not new_claims:
            return mask.copy()

        return _resolve_overlaps_gpu(new_claims, mask_gpu)


def apply_morphological_closing(
    mask: np.ndarray, radius: int = 5, max_threads: int = 1,
) -> np.ndarray:
    """Apply per-label morphological closing with overlap resolution.

    Parameters
    ----------
    mask : np.ndarray
        3D segmentation mask (Z, Y, X), integer labels.
    radius : int
        Ball radius for structuring element. 0 = no-op.
    max_threads : int
        Number of threads for parallel label closing. scipy binary_closing
        releases the GIL, so threads provide real speedup without memory copies.

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

    background = mask == 0

    def _close_one(label):
        binary_mask = mask == label
        closed = binary_closing(binary_mask, structure=struct)
        newly_claimed = closed & background
        if newly_claimed.any():
            return (label, newly_claimed)
        return None

    new_claims = {}
    if max_threads <= 1:
        for label in tqdm(labels, desc=f"Closing (r={radius})", leave=False):
            result = _close_one(label)
            if result is not None:
                new_claims[result[0]] = result[1]
    else:
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            futures = {pool.submit(_close_one, l): l for l in labels}
            for f in tqdm(as_completed(futures), total=len(labels),
                          desc=f"Closing (r={radius})", leave=False):
                result = f.result()
                if result is not None:
                    new_claims[result[0]] = result[1]

    if not new_claims:
        return mask.copy()

    return _resolve_overlaps(new_claims, mask)


def extract_clone_mask(
    mask: np.ndarray,
    cbox: BoxCoords,
    bbox: BoxCoords,
    mask_space: str,
    closing_radius: int = 5,
    max_threads: int = 1,
    use_gpu: bool = False,
) -> tuple[np.ndarray, dict]:
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
    tuple[np.ndarray, dict]
        B-box-sized mask with clone region populated, and stats dict
        with keys "n_labels" and "n_voxels" computed on the small crop.
    """
    if mask_space == "whole_brain":
        clone_crop = mask[cbox.to_slices()].copy()
        rel_cbox = compute_relative_cbox(cbox, bbox)
    else:
        rel_cbox = compute_relative_cbox(cbox, bbox)
        clone_crop = mask[rel_cbox.to_slices()].copy()

    if closing_radius > 0:
        if use_gpu and HAS_GPU:
            clone_crop = apply_morphological_closing_gpu(
                clone_crop, radius=closing_radius,
            )
        else:
            clone_crop = apply_morphological_closing(
                clone_crop, radius=closing_radius, max_threads=max_threads,
            )

    # Compute stats on the small crop before embedding into full output
    stats = {
        "n_labels": len(np.unique(clone_crop)) - (1 if 0 in clone_crop else 0),
        "n_voxels": int(np.count_nonzero(clone_crop)),
    }

    output = np.zeros(bbox.shape(), dtype=mask.dtype)
    output[rel_cbox.to_slices()] = clone_crop

    return output, stats


def process_brain(
    spec: BrainCloneSpec,
    output_dir: str,
    closing_radius: int = 5,
    date_tag: str = "0129",
    naming_template: str = "{brain_base}_{clone_name}_useg_{date}_cp_masks.tif",
    max_threads: int = 1,
    use_gpu: bool = False,
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

    # Handle 2x Z resolution: if mask Z is ~2x B-box Z, scale B-box Z coordinates
    bbox_shape = spec.bbox.shape()
    z_ratio = mask.shape[0] / bbox_shape[0]
    if 1.9 <= z_ratio <= 2.1:  # Allow for small rounding differences
        print(f"  Detected 2x Z resolution (ratio={z_ratio:.2f}): scaling B-box Z from {bbox_shape[0]} to match mask Z={mask.shape[0]}")
        spec.bbox.zmin *= 2
        spec.bbox.zmax = min(spec.bbox.zmax * 2, mask.shape[0])

    stem = Path(spec.mask_path).stem
    brain_base = stem[:-len("_cp_masks")] if stem.endswith("_cp_masks") else stem

    output_paths = []
    for clone_name, cbox in spec.clones.items():
        print(f"\n  Processing {clone_name}...")
        t0 = time.time()

        result, stats = extract_clone_mask(
            mask, cbox, spec.bbox,
            mask_space=mask_space,
            closing_radius=closing_radius,
            max_threads=max_threads,
            use_gpu=use_gpu,
        )

        n_labels = stats["n_labels"]
        n_voxels = stats["n_voxels"]
        elapsed = time.time() - t0
        print(f"    Cells: {n_labels}, Labeled voxels: {n_voxels}, Time: {elapsed:.1f}s")

        filename = naming_template.format(
            brain_base=brain_base,
            clone_name=clone_name,
            date=date_tag,
        )
        out_path = str(out_dir / filename)
        tifffile.imwrite(out_path, result.astype(np.uint16), compression="zstd")
        print(f"    Saved: {filename}")
        output_paths.append(out_path)

    return output_paths


def run_pipeline(config: CloneExtractConfig, workers: int = 1, use_gpu: bool = False) -> list[str]:
    """Run the full clone extraction pipeline.

    Parameters
    ----------
    config : CloneExtractConfig
        Pipeline configuration.
    workers : int
        Number of parallel brain workers. Each worker also uses threads
        internally for label-level closing parallelism.
    use_gpu : bool
        Use GPU-accelerated morphological closing via CuPy.
    """
    if use_gpu and not HAS_GPU:
        print("WARNING: --gpu requested but CuPy is not available. Falling back to CPU.")
        use_gpu = False

    max_threads = max(1, os.cpu_count() // max(workers, 1))
    max_threads = min(max_threads, 8)

    common_kwargs = dict(
        output_dir=config.output_dir,
        closing_radius=config.closing_radius,
        date_tag=config.date_tag,
        naming_template=config.naming_template,
        max_threads=max_threads,
        use_gpu=use_gpu,
    )

    all_paths: list[str] = []
    if workers <= 1:
        for spec in config.brains:
            paths = process_brain(spec, **common_kwargs)
            all_paths.extend(paths)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_brain, spec, **common_kwargs): spec.brain_name
                for spec in config.brains
            }
            for f in as_completed(futures):
                brain_name = futures[f]
                try:
                    all_paths.extend(f.result())
                except Exception as exc:
                    print(f"Brain {brain_name} failed: {exc}")
                    raise

    return all_paths
