"""Coordinate transformations for 3D clone unrolling.

This module implements the unrolling transformation using rotation matrices
to preserve cell shapes (rigid transformation).

The transformation works by:
1. Computing rotation matrices to align each segment with the reference axis
2. Transforming anchor points to lie along a straight line (reference axis direction)
3. Applying the same rotation to cells/voxels relative to their anchor
"""

import numpy as np
from typing import Union


def compute_rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that rotates source vector to align with target vector.

    Uses Rodrigues' rotation formula. Handles edge cases where vectors are
    parallel or anti-parallel.

    Args:
        source: Source direction vector (will be normalized).
        target: Target direction vector (will be normalized).

    Returns:
        3x3 rotation matrix R such that R @ source is parallel to target.
    """
    # Normalize vectors
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)

    if source_norm < 1e-10 or target_norm < 1e-10:
        # Degenerate case - return identity
        return np.eye(3)

    source = source / source_norm
    target = target / target_norm

    # Compute cross product and dot product
    cross = np.cross(source, target)
    dot = np.dot(source, target)

    # Check if vectors are parallel (dot ≈ 1)
    if dot > 1 - 1e-10:
        return np.eye(3)

    # Check if vectors are anti-parallel (dot ≈ -1)
    if dot < -1 + 1e-10:
        # Need 180° rotation around any perpendicular axis
        # Find a perpendicular vector
        if abs(source[0]) < 0.9:
            perp = np.cross(source, np.array([1, 0, 0]))
        else:
            perp = np.cross(source, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        # 180° rotation around perp: R = 2 * perp @ perp.T - I
        return 2 * np.outer(perp, perp) - np.eye(3)

    # Rodrigues' rotation formula
    # R = I + [v]_x + [v]_x^2 * (1 - dot) / ||v||^2
    # where v = source x target, [v]_x is skew-symmetric matrix
    v = cross
    v_norm_sq = np.dot(v, v)

    # Skew-symmetric matrix
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * (1 - dot) / v_norm_sq

    return R


def unroll_clone(
    centroids: dict,
    anchors: np.ndarray,
    assignments: dict,
) -> tuple:
    """Perform full unrolling transformation on cell centroids.

    Unrolls the clone along the start→end direction (reference axis).
    Uses rotation matrices to preserve cell shapes.

    Args:
        centroids: Dictionary mapping cell_id to centroid coordinates [z, y, x].
        anchors: Array of anchor points, shape (n_anchors, 3).
        assignments: Dictionary mapping cell_id to anchor index.

    Returns:
        Tuple of:
            - transformed_centroids: dict mapping cell_id to transformed coordinates
            - transform_params: dict containing all transformation parameters
              needed to apply the same transform to other data (e.g., puncta)
    """
    n_anchors = len(anchors)

    # Reference axis: direction from first anchor (start) to last anchor (end)
    ref_axis = anchors[-1] - anchors[0]
    ref_axis_length = np.linalg.norm(ref_axis)
    if ref_axis_length < 1e-10:
        raise ValueError("Start and end anchors are at the same position")
    ref_axis_normalized = ref_axis / ref_axis_length

    # Transform anchor points to lie along the reference axis direction
    # anchor_transformed[0] = anchors[0] (start stays in place)
    # anchor_transformed[i+1] = anchor_transformed[i] + segment_length * ref_axis_normalized
    anchors_transformed = np.zeros_like(anchors)
    anchors_transformed[0] = anchors[0].copy()  # Start anchor stays in place

    # Store rotation matrices for each segment
    rotation_matrices = []

    for i in range(n_anchors - 1):
        # Compute segment vector and its length
        segment_vec = anchors[i + 1] - anchors[i]
        segment_length = np.linalg.norm(segment_vec)

        if segment_length < 1e-10:
            # Degenerate segment - use identity rotation
            R = np.eye(3)
        else:
            # Compute rotation matrix to align segment with reference axis
            segment_dir = segment_vec / segment_length
            R = compute_rotation_matrix(segment_dir, ref_axis_normalized)

        rotation_matrices.append(R)

        # Transform next anchor: move along reference axis by segment length
        anchors_transformed[i + 1] = anchors_transformed[i] + segment_length * ref_axis_normalized

    # Transform cell centroids
    transformed_centroids = {}
    cell_transform_info = {}

    for cell_id, centroid in centroids.items():
        anchor_idx = assignments.get(cell_id)
        if anchor_idx is None:
            # Cell not assigned - skip
            continue

        # Get the rotation matrix for this cell's segment
        # Cells assigned to anchor i use segment (i, i+1), except for the last anchor
        segment_idx = min(anchor_idx, n_anchors - 2)
        R = rotation_matrices[segment_idx]

        # Compute offset from anchor
        offset = centroid - anchors[anchor_idx]

        # Apply rotation to offset
        offset_rotated = R @ offset

        # Compute transformed centroid
        transformed = anchors_transformed[anchor_idx] + offset_rotated
        transformed_centroids[cell_id] = transformed

        # Store per-cell transformation info
        cell_transform_info[str(cell_id)] = {
            "centroid_original": centroid.tolist(),
            "centroid_transformed": transformed.tolist(),
            "anchor_index": anchor_idx,
            "rotation_matrix": R.tolist(),
        }

    # Build transform_params dict for saving
    transform_params = {
        "anchors_original": anchors.tolist(),
        "anchors_transformed": anchors_transformed.tolist(),
        "reference_axis": ref_axis_normalized.tolist(),
        "rotation_matrices": [R.tolist() for R in rotation_matrices],
        "cells": cell_transform_info,
    }

    return transformed_centroids, transform_params


def apply_transform_to_points(
    points: np.ndarray,
    mask: np.ndarray,
    transform_params: dict,
) -> tuple:
    """Apply saved transformation to arbitrary points (e.g., puncta).

    For each point:
    1. Look up which cell it belongs to (from mask)
    2. Get that cell's transformation parameters
    3. Apply the same transformation (rotation + translation)

    Args:
        points: Array of shape (N, 3) with coordinates in [z, y, x] order.
        mask: 3D label mask to determine which cell each point belongs to.
        transform_params: Dictionary from unroll_clone containing transformation info.

    Returns:
        Tuple of:
            - transformed_points: Array of shape (N, 3) with transformed coordinates
            - cell_assignments: Array of shape (N,) with cell IDs for each point
              (0 if point is in background)
    """
    n_points = len(points)
    transformed_points = np.full((n_points, 3), np.nan)
    cell_assignments = np.zeros(n_points, dtype=np.int32)

    cells_info = transform_params["cells"]

    for i, point in enumerate(points):
        # Get integer coordinates for mask lookup
        z, y, x = int(round(point[0])), int(round(point[1])), int(round(point[2]))

        # Check bounds
        if not (0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]):
            continue

        # Look up cell ID from mask
        cell_id = mask[z, y, x]
        cell_assignments[i] = cell_id

        if cell_id == 0:
            # Background - skip
            continue

        cell_key = str(cell_id)
        if cell_key not in cells_info:
            # Cell not in transform - skip
            continue

        cell_info = cells_info[cell_key]
        centroid_orig = np.array(cell_info["centroid_original"])
        centroid_trans = np.array(cell_info["centroid_transformed"])
        R = np.array(cell_info["rotation_matrix"])

        # Compute point's offset from cell centroid
        offset = point - centroid_orig

        # Apply rotation
        offset_rotated = R @ offset

        # Add to transformed cell centroid
        transformed_points[i] = centroid_trans + offset_rotated

    return transformed_points, cell_assignments


def transform_mask(
    mask: np.ndarray,
    transform_params: dict,
    output_shape: Union[tuple, None] = None,
    padding: int = 50,
) -> tuple:
    """Transform a 3D label mask using the unrolling transformation.

    Uses forward mapping: for each foreground voxel in the original mask,
    compute its transformed position and write to the output.

    Args:
        mask: 3D label mask, shape (Z, Y, X).
        transform_params: Dictionary from unroll_clone.
        output_shape: Shape of output mask. If None, auto-calculated from
            transformed coordinates with padding.
        padding: Padding to add around transformed coordinates when
            auto-calculating output shape.

    Returns:
        Tuple of:
            - transformed_mask: 3D label mask in unrolled space
            - output_offset: [z, y, x] offset of output origin from (0,0,0)
    """
    cells_info = transform_params["cells"]

    # First pass: determine output bounds from transformed cell centroids
    all_transformed = []
    for cell_info in cells_info.values():
        all_transformed.append(cell_info["centroid_transformed"])
    all_transformed = np.array(all_transformed)

    if output_shape is None:
        # Calculate bounds
        min_coords = np.floor(all_transformed.min(axis=0)).astype(int) - padding
        max_coords = np.ceil(all_transformed.max(axis=0)).astype(int) + padding

        output_shape = tuple(max_coords - min_coords)
        output_offset = min_coords
    else:
        output_offset = np.zeros(3, dtype=int)

    # Initialize output mask
    transformed_mask = np.zeros(output_shape, dtype=mask.dtype)

    # Get all foreground voxel coordinates
    foreground_coords = np.argwhere(mask > 0)

    if len(foreground_coords) == 0:
        return transformed_mask, output_offset

    # Get cell labels for each voxel
    voxel_labels = mask[foreground_coords[:, 0], foreground_coords[:, 1], foreground_coords[:, 2]]

    # Transform voxels cell by cell
    unique_labels = np.unique(voxel_labels)

    for label in unique_labels:
        cell_key = str(label)
        if cell_key not in cells_info:
            continue

        cell_info = cells_info[cell_key]
        centroid_orig = np.array(cell_info["centroid_original"])
        centroid_trans = np.array(cell_info["centroid_transformed"])
        R = np.array(cell_info["rotation_matrix"])

        # Get voxels belonging to this cell
        cell_mask = voxel_labels == label
        cell_voxels = foreground_coords[cell_mask].astype(float)

        # Compute offsets from cell centroid
        offsets = cell_voxels - centroid_orig

        # Apply rotation to all offsets at once (vectorized)
        # offsets is (N, 3), R is (3, 3)
        # We want R @ each offset, which is (offsets @ R.T)
        offsets_rotated = offsets @ R.T

        # Compute transformed voxel positions
        transformed_voxels = centroid_trans + offsets_rotated

        # Adjust for output offset
        transformed_voxels_adjusted = transformed_voxels - output_offset

        # Round to integer coordinates
        transformed_int = np.round(transformed_voxels_adjusted).astype(int)

        # Write to output mask (check bounds)
        for j, (tz, ty, tx) in enumerate(transformed_int):
            if (0 <= tz < output_shape[0] and
                0 <= ty < output_shape[1] and
                0 <= tx < output_shape[2]):
                transformed_mask[tz, ty, tx] = label

    return transformed_mask, output_offset


# Keep old functions for backward compatibility but mark as deprecated
def cart2sph(xyz: np.ndarray) -> np.ndarray:
    """DEPRECATED: Convert Cartesian coordinates to spherical coordinates.

    This function is kept for backward compatibility but should not be used
    for the unrolling transformation as it does not preserve distances.
    """
    xyz = np.atleast_2d(xyz)
    z, y, x = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    phi = np.zeros_like(r)
    nonzero_r = r > 1e-10
    phi[nonzero_r] = np.arccos(np.clip(z[nonzero_r] / r[nonzero_r], -1, 1))

    xy_norm = np.sqrt(x**2 + y**2)
    theta = np.zeros_like(r)
    nonzero_xy = xy_norm > 1e-10
    theta[nonzero_xy] = np.arccos(
        np.clip(x[nonzero_xy] / xy_norm[nonzero_xy], -1, 1)
    ) * np.sign(y[nonzero_xy])
    y_zero_x_neg = (np.abs(y) < 1e-10) & (x < 0)
    theta[y_zero_x_neg] = np.pi

    result = np.column_stack([r, phi, theta])
    return result.squeeze() if xyz.shape[0] == 1 else result


def sph2cart(rpt: np.ndarray) -> np.ndarray:
    """DEPRECATED: Convert spherical coordinates to Cartesian coordinates.

    This function is kept for backward compatibility but should not be used
    for the unrolling transformation as it does not preserve distances.
    """
    rpt = np.atleast_2d(rpt)
    r, phi, theta = rpt[:, 0], rpt[:, 1], rpt[:, 2]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    result = np.column_stack([z, y, x])
    return result.squeeze() if rpt.shape[0] == 1 else result
