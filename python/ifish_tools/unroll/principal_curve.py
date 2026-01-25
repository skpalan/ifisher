"""Principal curve computation for 3D clone unrolling.

This module provides functions to:
1. Compute cell centroids from a 3D label mask
2. Fit a principal curve through cell centroids using user-defined endpoints
3. Assign cells to anchor points along the curve
"""

import numpy as np
from scipy import ndimage


def compute_centroids(mask: np.ndarray) -> dict[int, np.ndarray]:
    """Compute the centroid (center of mass) for each labeled region in a 3D mask.

    Args:
        mask: 3D numpy array with integer labels. Label 0 is background.

    Returns:
        Dictionary mapping label ID (int) to centroid coordinates (numpy array
        of shape (3,) in [z, y, x] order).
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]  # Remove background

    # Use vectorized center_of_mass for all labels at once (much faster)
    centroids_list = ndimage.center_of_mass(mask, mask, labels)
    centroids = {
        int(label): np.array(com) for label, com in zip(labels, centroids_list)
    }

    return centroids


def fit_principal_curve(
    centroids: dict[int, np.ndarray],
    start_cell_id: int,
    end_cell_id: int,
    n_anchors: int = 30,
) -> tuple[np.ndarray, dict[int, int]]:
    """Fit a principal curve through cell centroids using endpoint-guided projection.

    The algorithm:
    1. Anchor[0] = start_cell centroid (contains ONLY the start cell)
    2. Project all other cells onto the start→end axis
    3. Sort cells by projection and divide into (n_anchors - 1) equal-count bins
    4. Anchor[i] = mean centroid of cells in bin i (for i = 1 to n_anchors-1)
    5. Assign each cell to its NEAREST anchor (Euclidean distance)

    This ensures:
    - Dense regions get more anchors (equal cells per anchor)
    - Start cell (neuroblast) is isolated in anchor 0
    - End cell naturally falls into the last bin with neighbors

    Args:
        centroids: Dictionary mapping cell IDs to centroid coordinates [z, y, x].
        start_cell_id: Label ID of the starting cell (e.g., neuroblast).
        end_cell_id: Label ID of the ending cell.
        n_anchors: Number of anchor points to generate (default 30).

    Returns:
        Tuple of:
            - anchors: numpy array of shape (n_anchors, 3) with anchor coordinates
            - assignments: dict mapping cell_id to anchor index (0 to n_anchors-1)

    Raises:
        ValueError: If start_cell_id or end_cell_id not found in centroids.
    """
    if start_cell_id not in centroids:
        raise ValueError(f"start_cell_id {start_cell_id} not found in centroids")
    if end_cell_id not in centroids:
        raise ValueError(f"end_cell_id {end_cell_id} not found in centroids")

    start_point = centroids[start_cell_id].copy()
    end_point = centroids[end_cell_id].copy()

    # Principal axis vector
    axis = end_point - start_point
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-10:
        raise ValueError("Start and end cells have the same centroid")

    # Initialize anchors array
    anchors = np.zeros((n_anchors, 3))

    # Anchor 0: ONLY the start cell (neuroblast)
    anchors[0] = start_point

    # Get all cells except the start cell
    other_cell_ids = [cid for cid in centroids.keys() if cid != start_cell_id]
    other_centroids = np.array([centroids[cid] for cid in other_cell_ids])

    # Project other cells onto the axis
    # t = dot(c - start, axis) / dot(axis, axis)
    # t=0 at start, t=1 at end
    projections = np.dot(other_centroids - start_point, axis) / np.dot(axis, axis)

    # Sort cells by projection value
    sorted_indices = np.argsort(projections)
    sorted_cell_ids = [other_cell_ids[i] for i in sorted_indices]
    sorted_centroids = other_centroids[sorted_indices]

    # Divide sorted cells into (n_anchors - 1) bins with equal cell counts
    n_other_cells = len(sorted_cell_ids)
    n_bins = n_anchors - 1  # Anchors 1 to n_anchors-1

    cells_per_bin = n_other_cells / n_bins

    # Compute anchor positions for anchors 1 to n_anchors-1
    for i in range(n_bins):
        bin_start_idx = int(i * cells_per_bin)
        bin_end_idx = int((i + 1) * cells_per_bin)

        # Last bin gets all remaining cells
        if i == n_bins - 1:
            bin_end_idx = n_other_cells

        # Get cells in this bin
        bin_centroids = sorted_centroids[bin_start_idx:bin_end_idx]

        if len(bin_centroids) == 0:
            # Empty bin (shouldn't happen with this algorithm, but just in case)
            # Interpolate position along the axis
            t = (i + 1) / n_bins
            anchors[i + 1] = start_point + t * axis
        else:
            # Anchor is mean centroid of cells in this bin
            anchors[i + 1] = np.mean(bin_centroids, axis=0)

    # Assign each cell to its NEAREST anchor (Euclidean distance)
    # Exception: start_cell is always assigned to anchor 0
    assignments = {}
    assignments[start_cell_id] = 0

    for cell_id in other_cell_ids:
        centroid = centroids[cell_id]
        distances = np.linalg.norm(anchors - centroid, axis=1)
        nearest_anchor = int(np.argmin(distances))

        # Cells should not be assigned to anchor 0 (reserved for neuroblast)
        # If nearest is anchor 0, assign to anchor 1 instead
        if nearest_anchor == 0:
            nearest_anchor = 1

        assignments[cell_id] = nearest_anchor

    return anchors, assignments


def assign_cells_to_anchors(
    centroids: dict[int, np.ndarray], anchors: np.ndarray
) -> dict[int, int]:
    """Assign each cell to its nearest anchor point.

    This is an alternative to the bin-based assignment in fit_principal_curve,
    useful when you want to reassign cells to anchors based on Euclidean distance.

    Args:
        centroids: Dictionary mapping cell IDs to centroid coordinates.
        anchors: numpy array of shape (n_anchors, 3) with anchor coordinates.

    Returns:
        Dictionary mapping cell_id to anchor index.
    """
    assignments = {}

    for cell_id, centroid in centroids.items():
        # Find nearest anchor
        distances = np.linalg.norm(anchors - centroid, axis=1)
        nearest_anchor = np.argmin(distances)
        assignments[cell_id] = int(nearest_anchor)

    return assignments
