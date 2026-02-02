"""Coordinate transformations for 3D clone unrolling.

Implements the spherical-coordinate-based unrolling from the TopoVelo neural tube
analysis for centroid positioning, combined with rigid rotation matrices
(Rodrigues) for shape-preserving mask/puncta transformation.

The transformation works by:
1. Centroid positions: spherical angle subtraction removes curvature (TopoVelo)
2. Mask voxels / puncta: rigid rotation relative to cell centroid preserves shape
"""

import numpy as np
from typing import Union


def cart2spherical(x, y, z):
    """Convert Cartesian to spherical coordinates.

    Args:
        x, y, z: Arrays of Cartesian coordinates.

    Returns:
        Tuple of (r, phi, theta) where:
            r = radial distance
            phi = polar angle (0 to pi)
            theta = azimuthal angle (-pi to pi)
    """
    x, y, z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(np.clip(z / (r + (r == 0)), -1, 1))
    theta = np.arctan2(y, x)
    return r, phi, theta


def spherical2cart(r, phi, theta):
    """Convert spherical to Cartesian coordinates.

    Args:
        r: Radial distance.
        phi: Polar angle (0 to pi).
        theta: Azimuthal angle (-pi to pi).

    Returns:
        Tuple of (x, y, z) Cartesian coordinates.
    """
    r, phi, theta = np.asarray(r, dtype=float), np.asarray(phi, dtype=float), np.asarray(theta, dtype=float)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


# Keep old names for backward compatibility
cart2sph = cart2spherical
sph2cart = spherical2cart


def compute_rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that rotates source vector to align with target vector.

    Uses Rodrigues' rotation formula.

    Args:
        source: Source direction vector (will be normalized).
        target: Target direction vector (will be normalized).

    Returns:
        3x3 rotation matrix R such that R @ source is parallel to target.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)

    if source_norm < 1e-10 or target_norm < 1e-10:
        return np.eye(3)

    source = source / source_norm
    target = target / target_norm

    cross = np.cross(source, target)
    dot = np.dot(source, target)

    if dot > 1 - 1e-10:
        return np.eye(3)

    if dot < -1 + 1e-10:
        if abs(source[0]) < 0.9:
            perp = np.cross(source, np.array([1, 0, 0]))
        else:
            perp = np.cross(source, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)

    v = cross
    v_norm_sq = np.dot(v, v)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * (1 - dot) / v_norm_sq
    return R


def unroll_clone(
    centroids: dict[int, np.ndarray],
    anchors_ordered: np.ndarray,
    anchor_positions: np.ndarray,
    assignments: dict[int, int],
    start_cell_id: int,
    plane: str = "zx",
) -> tuple[dict, np.ndarray, dict]:
    """Unroll a 3D clone using spherical coordinate transformation.

    Follows the TopoVelo curve2tube approach for centroid positioning.
    Also computes rigid rotation matrices per segment for shape-preserving
    mask/puncta transformation.

    After transformation, coordinates are shifted so the neuroblast (start_cell)
    centroid is at the origin along the principal axis.

    Args:
        centroids: Dict mapping cell_id to centroid [z, y, x].
        anchors_ordered: Array of anchor indices in traversal order.
        anchor_positions: (n_anchors, 3) original anchor coordinates.
        assignments: Dict mapping cell_id to anchor index.
        start_cell_id: Cell ID of the neuroblast (for origin offset).
        plane: Unrolling plane ('xy', 'yz', or 'zx').

    Returns:
        Tuple of:
            - transformed_centroids: dict cell_id -> (3,) transformed coords
            - curve_trans: (n_anchors, 3) transformed anchor positions
            - transform_params: dict with all info needed to transform masks/puncta
    """
    n_anchors = len(anchors_ordered)
    ax = anchor_positions[:, 0]
    ay = anchor_positions[:, 1]
    az = anchor_positions[:, 2]

    cell_ids = list(centroids.keys())

    # Reference axis: overall start→end direction (for Rodrigues rotation)
    start_pos = anchor_positions[anchors_ordered[0]]
    end_pos = anchor_positions[anchors_ordered[-1]]
    ref_axis = end_pos - start_pos
    ref_axis_norm = np.linalg.norm(ref_axis)
    if ref_axis_norm < 1e-10:
        raise ValueError("Start and end anchors are at the same position")
    ref_axis_normalized = ref_axis / ref_axis_norm

    # --- Pass 1: Spherical unrolling for centroids (TopoVelo) ---
    transformed_centroids = {}
    curve_trans = np.zeros((n_anchors, 3))
    anchor_transforms = []

    anchor_trans = np.array([0.0, 0.0, 0.0])
    r_v = phi_v = theta_v = None

    # Also compute rotation matrices per segment (for mask/puncta)
    rotation_matrices = []

    for i, anchor_idx in enumerate(anchors_ordered):
        curve_trans[i] = anchor_trans.copy()

        # Direction to next anchor
        if i < n_anchors - 1:
            next_idx = anchors_ordered[i + 1]
            segment_vec = anchor_positions[next_idx] - anchor_positions[anchor_idx]
            segment_length = np.linalg.norm(segment_vec)

            dz = segment_vec[0]
            dy = segment_vec[1]
            dx = segment_vec[2]
            r_v, phi_v, theta_v = cart2spherical(
                np.array([dz]), np.array([dy]), np.array([dx])
            )

            # Rodrigues rotation: align segment direction to reference axis
            if segment_length > 1e-10:
                R = compute_rotation_matrix(segment_vec / segment_length, ref_axis_normalized)
            else:
                R = np.eye(3)
        else:
            R = rotation_matrices[-1] if rotation_matrices else np.eye(3)

        rotation_matrices.append(R)

        # Store per-anchor transform info
        anchor_transforms.append({
            "anchor_index": int(anchor_idx),
            "anchor_original": anchor_positions[anchor_idx].tolist(),
            "anchor_transformed": anchor_trans.tolist(),
            "rotation_matrix": R.tolist(),
            "phi_v": float(phi_v[0]) if phi_v is not None else 0.0,
            "theta_v": float(theta_v[0]) if theta_v is not None else 0.0,
            "plane": plane,
        })

        # Transform centroids assigned to this anchor (spherical method)
        cells_at_anchor = [
            cid for cid, aidx in assignments.items() if aidx == anchor_idx
        ]

        if cells_at_anchor:
            cell_centroids = np.array([centroids[cid] for cid in cells_at_anchor])
            dz = cell_centroids[:, 0] - ax[anchor_idx]
            dy = cell_centroids[:, 1] - ay[anchor_idx]
            dx = cell_centroids[:, 2] - az[anchor_idx]

            r, phi, theta = cart2spherical(dz, dy, dx)

            if plane == "xy":
                xi, yi, zi = spherical2cart(r, phi - phi_v, theta)
            elif plane == "yz":
                xi, yi, zi = spherical2cart(
                    r, phi + (np.pi / 2 - phi_v), theta - theta_v
                )
            elif plane == "zx":
                xi, yi, zi = spherical2cart(
                    r, phi + (np.pi / 2 - phi_v), theta + (np.pi / 2 - theta_v)
                )
            else:
                raise ValueError(f"Unknown plane: {plane}")

            for j, cid in enumerate(cells_at_anchor):
                transformed_centroids[cid] = np.array([
                    xi[j] + anchor_trans[0],
                    yi[j] + anchor_trans[1],
                    zi[j] + anchor_trans[2],
                ])

        # Advance anchor translation
        if i < n_anchors - 1:
            if plane == "xy":
                delta = spherical2cart(r_v, np.array([0.0]), np.array([theta_v[0]]))
            elif plane == "yz":
                delta = spherical2cart(r_v, np.array([np.pi / 2]), np.array([0.0]))
            elif plane == "zx":
                delta = spherical2cart(
                    r_v, np.array([np.pi / 2]), np.array([np.pi / 2])
                )
            anchor_trans = anchor_trans + np.array([
                float(delta[0]), float(delta[1]), float(delta[2])
            ])

    # --- Fix 2: Shift so neuroblast centroid is at the origin ---
    if start_cell_id in transformed_centroids:
        offset = transformed_centroids[start_cell_id].copy()
        for cid in transformed_centroids:
            transformed_centroids[cid] = transformed_centroids[cid] - offset
        curve_trans = curve_trans - offset
        for at in anchor_transforms:
            at["anchor_transformed"] = (np.array(at["anchor_transformed"]) - offset).tolist()

    # --- Build transform_params ---
    ordered_anchor_map = {
        int(anchors_ordered[i]): i for i in range(n_anchors)
    }

    cells_info = {}
    for cid in cell_ids:
        anchor_idx = assignments.get(cid)
        if anchor_idx is None or cid not in transformed_centroids:
            continue
        ordered_pos = ordered_anchor_map.get(anchor_idx, 0)
        cells_info[str(cid)] = {
            "centroid_original": centroids[cid].tolist(),
            "centroid_transformed": transformed_centroids[cid].tolist(),
            "anchor_index": int(anchor_idx),
            "ordered_anchor_pos": ordered_pos,
        }

    transform_params = {
        "anchors_original": anchor_positions.tolist(),
        "anchors_ordered": anchors_ordered.tolist(),
        "anchors_transformed": curve_trans.tolist(),
        "anchor_transforms": anchor_transforms,
        "plane": plane,
        "cells": cells_info,
    }

    return transformed_centroids, curve_trans, transform_params


def apply_transform_to_points(
    points: np.ndarray,
    mask: np.ndarray,
    transform_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved unrolling transformation to arbitrary points (e.g., puncta).

    Uses rigid rotation (Rodrigues) relative to cell centroid to preserve shapes.

    Args:
        points: Array of shape (N, 3) with coordinates in [z, y, x] order.
        mask: 3D label mask to determine which cell each point belongs to.
        transform_params: Dictionary from unroll_clone.

    Returns:
        Tuple of:
            - transformed_points: Array of shape (N, 3) with transformed coordinates
            - cell_assignments: Array of shape (N,) with cell IDs (0 = background)
    """
    n_points = len(points)
    transformed_points = np.full((n_points, 3), np.nan)
    cell_assignments = np.zeros(n_points, dtype=np.int32)

    cells_info = transform_params["cells"]
    anchor_transforms = transform_params["anchor_transforms"]

    for i, point in enumerate(points):
        z, y, x = int(round(point[0])), int(round(point[1])), int(round(point[2]))

        if not (0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]):
            continue

        cell_id = int(mask[z, y, x])
        cell_assignments[i] = cell_id

        if cell_id == 0:
            continue

        cell_key = str(cell_id)
        if cell_key not in cells_info:
            continue

        cell_info = cells_info[cell_key]
        centroid_orig = np.array(cell_info["centroid_original"])
        centroid_trans = np.array(cell_info["centroid_transformed"])

        ordered_pos = cell_info["ordered_anchor_pos"]
        R = np.array(anchor_transforms[ordered_pos]["rotation_matrix"])

        # Rigid transform: rotate offset from cell centroid, then translate
        offset = point - centroid_orig
        offset_rotated = R @ offset
        transformed_points[i] = centroid_trans + offset_rotated

    return transformed_points, cell_assignments


def transform_mask(
    mask: np.ndarray,
    transform_params: dict,
    output_shape: Union[tuple, None] = None,
    padding: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a 3D label mask using rigid rotation per cell.

    For each cell: rotates voxel offsets from cell centroid using the segment's
    rotation matrix, then translates to the transformed centroid. This preserves
    cell shapes (rigid body transformation).

    Args:
        mask: 3D label mask, shape (Z, Y, X).
        transform_params: Dictionary from unroll_clone.
        output_shape: Shape of output mask. If None, auto-calculated.
        padding: Padding around transformed coordinates.

    Returns:
        Tuple of:
            - transformed_mask: 3D label mask in unrolled space
            - output_offset: [z, y, x] offset of output origin
    """
    cells_info = transform_params["cells"]
    anchor_transforms = transform_params["anchor_transforms"]

    # Determine output bounds from transformed centroids
    all_transformed = np.array([
        ci["centroid_transformed"] for ci in cells_info.values()
    ])

    if output_shape is None:
        min_coords = np.floor(all_transformed.min(axis=0)).astype(int) - padding
        max_coords = np.ceil(all_transformed.max(axis=0)).astype(int) + padding
        output_shape = tuple(max_coords - min_coords)
        output_offset = min_coords
    else:
        output_offset = np.zeros(3, dtype=int)

    transformed_mask = np.zeros(output_shape, dtype=mask.dtype)

    foreground_coords = np.argwhere(mask > 0)
    if len(foreground_coords) == 0:
        return transformed_mask, output_offset

    voxel_labels = mask[
        foreground_coords[:, 0], foreground_coords[:, 1], foreground_coords[:, 2]
    ]

    unique_labels = np.unique(voxel_labels)

    for label in unique_labels:
        cell_key = str(label)
        if cell_key not in cells_info:
            continue

        cell_info = cells_info[cell_key]
        centroid_orig = np.array(cell_info["centroid_original"])
        centroid_trans = np.array(cell_info["centroid_transformed"])
        ordered_pos = cell_info["ordered_anchor_pos"]
        R = np.array(anchor_transforms[ordered_pos]["rotation_matrix"])

        # Get voxels for this cell
        cell_mask = voxel_labels == label
        cell_voxels = foreground_coords[cell_mask].astype(float)

        # Rigid transform: offset from centroid → rotate → translate
        offsets = cell_voxels - centroid_orig
        offsets_rotated = offsets @ R.T  # vectorized: (N,3) @ (3,3)
        transformed_voxels = centroid_trans + offsets_rotated

        # Write to output
        adjusted = transformed_voxels - output_offset
        int_coords = np.round(adjusted).astype(int)

        # Bounds check vectorized
        valid = (
            (int_coords[:, 0] >= 0) & (int_coords[:, 0] < output_shape[0]) &
            (int_coords[:, 1] >= 0) & (int_coords[:, 1] < output_shape[1]) &
            (int_coords[:, 2] >= 0) & (int_coords[:, 2] < output_shape[2])
        )
        valid_coords = int_coords[valid]
        transformed_mask[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = label

    return transformed_mask, output_offset
