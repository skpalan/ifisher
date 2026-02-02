"""Principal curve computation for 3D clone unrolling.

This module provides functions to:
1. Compute cell centroids from a 3D label mask
2. Fit an elastic principal curve through cell centroids using ElPiGraph
3. Sort anchor points along the curve from start to end
4. Assign cells to anchor points along the curve
"""

import numpy as np
from scipy import ndimage

import elpigraph


def compute_centroids(mask: np.ndarray) -> dict[int, np.ndarray]:
    """Compute the centroid (center of mass) for each labeled region in a 3D mask.

    Args:
        mask: 3D numpy array with integer labels. Label 0 is background.

    Returns:
        Dictionary mapping label ID (int) to centroid coordinates (numpy array
        of shape (3,) in [z, y, x] order).
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]

    centroids_list = ndimage.center_of_mass(mask, mask, labels)
    centroids = {
        int(label): np.array(com) for label, com in zip(labels, centroids_list)
    }

    return centroids


def fit_principal_curve(
    centroids: dict[int, np.ndarray],
    n_anchors: int = 30,
    epg_mu: float = 1.0,
    epg_lambda: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Fit an elastic principal curve through cell centroids using ElPiGraph.

    Args:
        centroids: Dictionary mapping cell IDs to centroid coordinates [z, y, x].
        n_anchors: Number of anchor (node) points on the curve.
        epg_mu: ElPiGraph stretching penalty.
        epg_lambda: ElPiGraph bending penalty.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - anchor_positions: (n_anchors, 3) node coordinates
            - edges: (n_edges, 2) edge list (adjacency)
            - node_degree: (n_anchors,) degree info from ElPiGraph (0 = endpoint)
            - assignments: dict mapping cell_id to nearest anchor index
    """
    cell_ids = list(centroids.keys())
    data = np.array([centroids[cid] for cid in cell_ids])

    result = elpigraph.computeElasticPrincipalCurve(
        data,
        NumNodes=n_anchors,
        Mu=epg_mu,
        Lambda=epg_lambda,
        verbose=False,
    )
    r = result[0]

    anchor_positions = r["NodePositions"]
    edges = r["Edges"][0]  # (n_edges, 2)
    node_degree = r["Edges"][2]  # 0 = endpoint

    # Hard assignment: each cell to its nearest anchor
    dists = np.linalg.norm(
        data[:, None, :] - anchor_positions[None, :, :], axis=2
    )
    hard_assign = np.argmin(dists, axis=1)
    assignments = {cid: int(hard_assign[i]) for i, cid in enumerate(cell_ids)}

    return anchor_positions, edges, node_degree, assignments


def sort_anchors(
    edges: np.ndarray,
    node_degree: np.ndarray,
    anchor_positions: np.ndarray,
    start_anchor: int,
    end_anchor: int,
) -> np.ndarray:
    """Order anchor points along the curve from start to end.

    Traverses the graph from start_anchor to end_anchor, producing an ordered
    sequence of anchor indices.

    Args:
        edges: (n_edges, 2) edge list.
        node_degree: (n_anchors,) degree info (0 = endpoint).
        anchor_positions: (n_anchors, 3) node coordinates.
        start_anchor: Index of the starting anchor (near neuroblast).
        end_anchor: Index of the ending anchor (far end).

    Returns:
        Array of anchor indices in traversal order from start to end.
    """
    n_nodes = len(anchor_positions)

    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for e in edges:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])

    # Traverse from start to end
    ordered = [start_anchor]
    prev = start_anchor
    ptr = start_anchor

    while len(ordered) < n_nodes:
        found_next = False
        for nb in adj[ptr]:
            if nb != prev:
                ordered.append(nb)
                prev = ptr
                ptr = nb
                found_next = True
                break
        if not found_next:
            break

    return np.array(ordered)


def assign_cells_to_anchors(
    centroids: dict[int, np.ndarray], anchors: np.ndarray
) -> dict[int, int]:
    """Assign each cell to its nearest anchor point.

    Args:
        centroids: Dictionary mapping cell IDs to centroid coordinates.
        anchors: numpy array of shape (n_anchors, 3) with anchor coordinates.

    Returns:
        Dictionary mapping cell_id to anchor index.
    """
    assignments = {}

    for cell_id, centroid in centroids.items():
        distances = np.linalg.norm(anchors - centroid, axis=1)
        nearest_anchor = np.argmin(distances)
        assignments[cell_id] = int(nearest_anchor)

    return assignments
