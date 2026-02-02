"""Unroll module for transforming 3D clones along their principal curve.

This module implements a method to "unroll" a 3D clone by:
1. Computing an elastic principal curve through cell centroids (ElPiGraph)
2. Transforming coordinates so the principal axis becomes straight
   (spherical coordinate rotation, following TopoVelo)

Based on the method described for neural tube unrolling using principal curves.
"""

from .principal_curve import (
    compute_centroids,
    fit_principal_curve,
    sort_anchors,
    assign_cells_to_anchors,
)
from .transform import (
    cart2spherical,
    spherical2cart,
    compute_rotation_matrix,
    unroll_clone,
    apply_transform_to_points,
    transform_mask,
    # Deprecated aliases
    cart2sph,
    sph2cart,
)
from .endpoints import (
    detect_endpoints,
    find_endpoint_anchors,
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
    "sort_anchors",
    "assign_cells_to_anchors",
    "cart2spherical",
    "spherical2cart",
    "unroll_clone",
    "apply_transform_to_points",
    "transform_mask",
    "detect_endpoints",
    "find_endpoint_anchors",
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
