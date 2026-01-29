"""
Label smoothing (diffusion) wrapper for refining segmentation boundaries.
"""

import numpy as np
import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D


def apply_label_diffusion(
    segmentation: np.ndarray,
    guide_image: np.ndarray,
    config: dict
) -> np.ndarray:
    """
    Apply label diffusion to smooth segmentation boundaries.
    
    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation mask (Z, Y, X)
    guide_image : np.ndarray
        3D guide image (Z, Y, X), normalized to 0-1
    config : dict
        Configuration dictionary with smoothing parameters
        
    Returns
    -------
    np.ndarray
        Smoothed 3D segmentation mask (Z, Y, X)
    """
    # Setup parameters
    params = uSegment3D_params.get_label_diffusion_params()
    
    # Apply user configuration
    params['diffusion']['refine_iters'] = config.get('refine_iters', 15)
    params['diffusion']['refine_alpha'] = config.get('refine_alpha', 0.60)
    params['diffusion']['refine_clamp'] = config.get('refine_clamp', 0.7)
    params['diffusion']['affinity_type'] = 'heat'
    params['diffusion']['noprogress_bool'] = True
    params['guide_img']['pmin'] = 0
    params['guide_img']['pmax'] = 100
    
    # Normalize guide image
    guide_normalized = guide_image.astype(np.float32)
    guide_normalized = (guide_normalized - guide_normalized.min()) / (guide_normalized.max() - guide_normalized.min())
    
    # Run label diffusion
    print("  Applying label diffusion smoothing...")
    segmentation_smoothed = uSegment3D.label_diffuse_3D_cell_segmentation_MP(
        segmentation,
        guide_image=guide_normalized,
        params=params
    )
    
    n_cells_before = len(np.unique(segmentation)) - 1
    n_cells_after = len(np.unique(segmentation_smoothed)) - 1
    print(f"    Cells: {n_cells_before} → {n_cells_after} (Δ{n_cells_after - n_cells_before:+d})")
    
    return segmentation_smoothed
