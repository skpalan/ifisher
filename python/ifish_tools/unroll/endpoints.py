"""Automatic endpoint detection for 3D clone unrolling.

Detects start (neuroblast = largest cell) and end (farthest cell from neuroblast)
automatically from a 3D segmentation mask.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_endpoints(
    mask: np.ndarray, centroids: dict[int, np.ndarray]
) -> tuple[int, int]:
    """Auto-detect start and end cells for unrolling.

    Start cell: label with the largest voxel count (neuroblast).
    End cell: label whose centroid is farthest from the start cell centroid.

    Args:
        mask: 3D label mask (Z, Y, X). Label 0 is background.
        centroids: Dictionary mapping label ID to centroid [z, y, x].

    Returns:
        Tuple of (start_cell_id, end_cell_id).
    """
    # Count voxels per label (skip background=0)
    counts = np.bincount(mask.ravel())
    counts[0] = 0  # ignore background
    start_cell_id = int(np.argmax(counts))

    logger.info(
        "Neuroblast (start): cell %d (%d voxels)", start_cell_id, counts[start_cell_id]
    )

    # Find cell farthest from start
    start_centroid = centroids[start_cell_id]
    max_dist = -1.0
    end_cell_id = start_cell_id

    for cid, centroid in centroids.items():
        if cid == start_cell_id:
            continue
        d = np.linalg.norm(centroid - start_centroid)
        if d > max_dist:
            max_dist = d
            end_cell_id = cid

    logger.info("Farthest cell (end): cell %d (dist=%.1f)", end_cell_id, max_dist)

    return start_cell_id, end_cell_id


def find_endpoint_anchors(
    anchor_positions: np.ndarray,
    node_degree: np.ndarray,
    start_centroid: np.ndarray,
    end_centroid: np.ndarray,
) -> tuple[int, int]:
    """Find which graph endpoints are nearest to the start/end cells.

    Args:
        anchor_positions: (n_anchors, 3) node coordinates.
        node_degree: (n_anchors,) degree info from ElPiGraph (0 = endpoint).
        start_centroid: Centroid of the start cell [z, y, x].
        end_centroid: Centroid of the end cell [z, y, x].

    Returns:
        Tuple of (start_anchor_idx, end_anchor_idx).
    """
    endpoint_indices = np.where(node_degree == 0)[0]

    if len(endpoint_indices) < 2:
        raise ValueError(
            f"Expected 2 graph endpoints, found {len(endpoint_indices)}. "
            "The principal curve may have formed a loop."
        )

    # Find which endpoint is closest to start cell
    dists_to_start = np.linalg.norm(
        anchor_positions[endpoint_indices] - start_centroid, axis=1
    )
    start_anchor = int(endpoint_indices[np.argmin(dists_to_start)])

    # End anchor is the other endpoint farthest from start anchor
    dists_to_end = np.linalg.norm(
        anchor_positions[endpoint_indices] - end_centroid, axis=1
    )
    end_anchor = int(endpoint_indices[np.argmin(dists_to_end)])

    # If they resolved to the same endpoint, pick the second-closest for end
    if start_anchor == end_anchor and len(endpoint_indices) > 1:
        sorted_end = endpoint_indices[np.argsort(dists_to_end)]
        end_anchor = int(sorted_end[1])

    return start_anchor, end_anchor
