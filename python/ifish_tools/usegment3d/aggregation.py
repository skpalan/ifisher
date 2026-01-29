"""
u-segment3D aggregation wrapper for combining 2D views into 3D segmentation.
"""

import numpy as np
import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D


def aggregate_direct_method(
    probs: dict,
    flows: dict,
    config: dict
) -> np.ndarray:
    """
    Run u-segment3D Direct Method to aggregate 2D views into 3D segmentation.
    
    Parameters
    ----------
    probs : dict
        Dictionary with keys 'xy', 'xz', 'yz' containing probability arrays
    flows : dict
        Dictionary with keys 'xy', 'xz', 'yz' containing flow gradient arrays
    config : dict
        Configuration dictionary with aggregation parameters
        
    Returns
    -------
    np.ndarray
        3D segmentation mask (Z, Y, X)
    """
    # Setup parameters
    params = uSegment3D_params.get_2D_to_3D_aggregation_params()
    
    # Apply user configuration
    params['postprocess_binary']['remove_small_objects'] = config.get('remove_small_objects', 1000)
    params['connected_component']['min_area'] = config.get('min_area', 500)
    params['combine_cell_probs']['min_prob_thresh'] = config.get('min_prob_thresh', 0.25)
    params['combine_cell_probs']['cellpose_prob_mask'] = True  # Using Cellpose probabilities
    params['gradient_descent']['gradient_decay'] = config.get('gradient_decay', 0.0)
    params['gradient_descent']['n_iter'] = config.get('n_iter', 200)
    params['gradient_descent']['do_mp'] = False
    params['gradient_descent']['debug_viz'] = False
    
    # Run aggregation
    print("  Running u-segment3D Direct Method aggregation...")
    segmentation_3d, (prob_3d, grad_3d) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(
        probs=[probs['xy'], probs['xz'], probs['yz']],
        gradients=[flows['xy'], flows['xz'], flows['yz']],
        params=params,
        savefolder=None,
        basename=None
    )
    
    n_cells = len(np.unique(segmentation_3d)) - 1
    print(f"    Detected {n_cells} cells")
    
    return segmentation_3d
