"""Unroll module for transforming 3D clones along their principal curve.

This module implements a method to "unroll" a 3D clone by:
1. Computing a principal curve through cell centroids
2. Transforming coordinates so the principal axis becomes straight

Based on the method described for neural tube unrolling using principal curves.
"""

from .principal_curve import (
    compute_centroids,
    fit_principal_curve,
    assign_cells_to_anchors,
)
from .transform import (
    compute_rotation_matrix,
    unroll_clone,
    apply_transform_to_points,
    transform_mask,
    # Deprecated but kept for backward compatibility
    cart2sph,
    sph2cart,
)
from .io import (
    load_mask,
    save_mask,
    load_transform,
    save_transform,
    load_puncta,
    save_puncta,
)

__all__ = [
    "compute_centroids",
    "fit_principal_curve",
    "assign_cells_to_anchors",
    "compute_rotation_matrix",
    "unroll_clone",
    "apply_transform_to_points",
    "transform_mask",
    "load_mask",
    "save_mask",
    "load_transform",
    "save_transform",
    "load_puncta",
    "save_puncta",
    # Deprecated
    "cart2sph",
    "sph2cart",
]
