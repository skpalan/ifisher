"""
Cellpose inference on orthogonal views (XY, XZ, YZ) for 3D images.
"""

import numpy as np
from tqdm import tqdm
from cellpose import models


def run_cellpose_on_view(
    image_3d: np.ndarray,
    model: models.CellposeModel,
    view: str = 'xy',
    channels: tuple[int, int] = (0, 0),
    diameter: float = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Cellpose slice-by-slice on a specific view of a 3D image.
    
    Parameters
    ----------
    image_3d : np.ndarray
        3D or 4D image array: (Z, Y, X) or (Z, Y, X, C)
    model : CellposeModel
        Loaded Cellpose model
    view : str
        Which view to process: 'xy', 'xz', or 'yz'
    channels : tuple[int, int]
        Cellpose channels [cyto, nucleus]
    diameter : float, optional
        Cell diameter (None = auto-detect)
    flow_threshold : float
        Flow threshold for masks
    cellprob_threshold : float
        Cell probability threshold
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (probs, flows) where:
        - probs: Cell probability (n_slices, H, W)
        - flows: Flow gradients (2, n_slices, H, W) in Y,X order
    """
    # Prepare image based on view
    # Supports both (Z, Y, X) and (Z, Y, X, C) inputs
    has_channels = image_3d.ndim == 4
    if view == 'xy':
        img_stack = image_3d  # (Z, Y, X[, C])
    elif view == 'xz':
        if has_channels:
            img_stack = image_3d.transpose(1, 0, 2, 3)  # (Y, Z, X, C)
        else:
            img_stack = image_3d.transpose(1, 0, 2)  # (Y, Z, X)
    elif view == 'yz':
        if has_channels:
            img_stack = image_3d.transpose(2, 0, 1, 3)  # (X, Z, Y, C)
        else:
            img_stack = image_3d.transpose(2, 0, 1)  # (X, Z, Y)
    else:
        raise ValueError(f"Invalid view: {view}. Must be 'xy', 'xz', or 'yz'")
    
    n_slices = img_stack.shape[0]
    probs_list = []
    flows_list = []
    
    # Process each slice
    for i in tqdm(range(n_slices), desc=f"  Cellpose {view.upper()}", leave=False):
        slice_2d = img_stack[i]
        
        # Run Cellpose
        masks, flows, styles = model.eval(
            slice_2d,
            channels=channels,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            compute_masks=True
        )
        
        # Extract probability and flow
        # flows is [dY, dX, cellprob, ...]
        probs_list.append(flows[2])  # Cell probability
        flows_list.append(flows[1])  # [dY, dX]
    
    # Stack results
    probs = np.stack(probs_list, axis=0)  # (n_slices, H, W)
    flows = np.stack(flows_list, axis=0)  # (n_slices, 2, H, W)
    
    return probs, flows


def run_cellpose_3_views(
    image_3d: np.ndarray,
    model: models.CellposeModel,
    channels: tuple[int, int] = (0, 0),
    diameter: float = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0
) -> dict:
    """
    Run Cellpose on all three orthogonal views of a 3D image.
    
    Parameters
    ----------
    image_3d : np.ndarray
        3D or 4D image array: (Z, Y, X) or (Z, Y, X, C)
    model : CellposeModel
        Loaded Cellpose model
    channels : tuple[int, int]
        Cellpose channels [cyto, nucleus]
    diameter : float, optional
        Cell diameter (None = auto-detect)
    flow_threshold : float
        Flow threshold for masks
    cellprob_threshold : float
        Cell probability threshold
        
    Returns
    -------
    dict
        Dictionary with keys 'probs' and 'flows', each containing:
        - 'xy': arrays for XY view
        - 'xz': arrays for XZ view
        - 'yz': arrays for YZ view
    """
    print("  Running Cellpose on 3 orthogonal views...")
    
    # XY view
    probs_xy_raw, flows_xy_raw = run_cellpose_on_view(
        image_3d, model, view='xy',
        channels=channels, diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    
    # Transpose to match u-segment3D expected format
    probs_xy = probs_xy_raw  # (Z, Y, X)
    flows_xy = flows_xy_raw.transpose(1, 0, 2, 3)  # (2, Z, Y, X)
    
    # XZ view
    probs_xz_raw, flows_xz_raw = run_cellpose_on_view(
        image_3d, model, view='xz',
        channels=channels, diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    
    # Transpose to match u-segment3D expected format
    probs_xz = probs_xz_raw.transpose(1, 0, 2)  # (Z, Y, X)
    flows_xz = flows_xz_raw.transpose(1, 2, 0, 3)  # (2, Z, Y, X)
    
    # YZ view
    probs_yz_raw, flows_yz_raw = run_cellpose_on_view(
        image_3d, model, view='yz',
        channels=channels, diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    
    # Transpose to match u-segment3D expected format
    probs_yz = probs_yz_raw.transpose(1, 2, 0)  # (Z, Y, X)
    flows_yz = flows_yz_raw.transpose(1, 2, 3, 0)  # (2, Z, Y, X)
    
    print(f"    XY probs: {probs_xy.shape}, flows: {flows_xy.shape}")
    print(f"    XZ probs: {probs_xz.shape}, flows: {flows_xz.shape}")
    print(f"    YZ probs: {probs_yz.shape}, flows: {flows_yz.shape}")
    
    return {
        'probs': {
            'xy': probs_xy,
            'xz': probs_xz,
            'yz': probs_yz
        },
        'flows': {
            'xy': flows_xy,
            'xz': flows_xz,
            'yz': flows_yz
        }
    }
