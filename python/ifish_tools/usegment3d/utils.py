"""
Utility functions for I/O, visualization, and mask manipulation.
"""

import os
import json
import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import skimage.segmentation as sksegmentation

from .config import BoundingBox


def combine_clone_masks(
    brain_shape: tuple[int, int, int],
    clone_data: list[tuple[BoundingBox, np.ndarray]]
) -> np.ndarray:
    """
    Combine multiple clone masks into a single full-brain mask.
    
    Handles label renumbering to avoid conflicts between clones.
    
    Parameters
    ----------
    brain_shape : tuple[int, int, int]
        Full brain dimensions (Z, Y, X)
    clone_data : list[tuple[BoundingBox, np.ndarray]]
        List of (bounding_box, segmentation_mask) tuples
        
    Returns
    -------
    np.ndarray
        Combined full-brain segmentation mask (Z, Y, X)
    """
    # Create empty full-brain mask
    full_mask = np.zeros(brain_shape, dtype=np.uint16)
    
    current_max_label = 0
    
    for bbox, clone_seg in clone_data:
        # Get slices for this clone
        z_slice, y_slice, x_slice = bbox.to_slices()
        
        # Renumber labels to avoid conflicts
        clone_seg_renumbered = clone_seg.copy()
        if current_max_label > 0:
            # Add offset to all non-zero labels
            mask_nonzero = clone_seg > 0
            clone_seg_renumbered[mask_nonzero] = clone_seg[mask_nonzero] + current_max_label
        
        # Place in full mask (only where full_mask is currently 0)
        region = full_mask[z_slice, y_slice, x_slice]
        mask_empty = region == 0
        region[mask_empty] = clone_seg_renumbered[mask_empty]
        full_mask[z_slice, y_slice, x_slice] = region
        
        # Update max label
        current_max_label = full_mask.max()
    
    return full_mask


def save_segmentation_mask(
    segmentation: np.ndarray,
    output_path: str
) -> None:
    """
    Save segmentation mask as TIFF file.
    
    Parameters
    ----------
    segmentation : np.ndarray
        Segmentation mask
    output_path : str
        Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, segmentation.astype(np.uint16), compression='zlib')


def create_visualization(
    segmentation: np.ndarray,
    guide_image: np.ndarray,
    z_slices: list[int],
    output_path: str
) -> None:
    """
    Create overlay visualization of segmentation on image at specified Z-slices.
    
    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation mask (Z, Y, X)
    guide_image : np.ndarray
        3D guide image (Z, Y, X)
    z_slices : list[int]
        Z-slice indices to visualize
    output_path : str
        Output PNG file path
    """
    n_slices = len(z_slices)
    fig, axes = plt.subplots(2, n_slices, figsize=(5*n_slices, 10))
    
    if n_slices == 1:
        axes = axes.reshape(2, 1)
    
    for i, z in enumerate(z_slices):
        if z >= segmentation.shape[0]:
            z = segmentation.shape[0] - 1
        
        # Original image
        axes[0, i].imshow(guide_image[z], cmap='gray')
        axes[0, i].set_title(f'Z={z} - Image')
        axes[0, i].axis('off')
        
        # Segmentation overlay
        overlay = sksegmentation.mark_boundaries(
            np.dstack([guide_image[z], guide_image[z], guide_image[z]]),
            segmentation[z],
            color=(0, 1, 0),
            mode='thick'
        )
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Z={z} - Segmentation')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def check_existing_outputs(
    output_dir: str,
    brain_name: str,
    brain_path: str,
    date: str,
    save_raw: bool = True,
    save_smoothed: bool = True
) -> tuple[bool, list[str]]:
    """
    Check if output files already exist for a brain.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    brain_name : str
        Brain name (e.g., 'brain08')
    brain_path : str
        Path to brain file (to extract base name)
    date : str
        Date string for filename
    save_raw : bool
        Whether raw output should exist
    save_smoothed : bool
        Whether smoothed output should exist
        
    Returns
    -------
    tuple[bool, list[str]]
        (all_exist, list_of_existing_files)
    """
    brain_base = Path(brain_path).stem
    
    expected_files = []
    if save_raw:
        raw_name = f"{brain_base}_useg_{date}_raw_cp_masks.tif"
        expected_files.append(os.path.join(output_dir, raw_name))
    
    if save_smoothed:
        smoothed_name = f"{brain_base}_useg_{date}_cp_masks.tif"
        expected_files.append(os.path.join(output_dir, smoothed_name))
    
    existing_files = [f for f in expected_files if os.path.exists(f)]
    all_exist = len(existing_files) == len(expected_files)
    
    return all_exist, existing_files


def save_processing_log(
    results: dict,
    output_path: str
) -> None:
    """
    Save processing results as JSON log.
    
    Parameters
    ----------
    results : dict
        Processing results with cell counts, timing, etc.
    output_path : str
        Output JSON file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Processing log saved to: {output_path}")
