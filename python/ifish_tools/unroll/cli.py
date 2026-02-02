"""Command-line interface for 3D tissue unrolling."""

import argparse
import logging
import re
import sys
from pathlib import Path

from .endpoints import detect_endpoints, find_endpoint_anchors
from .io import load_mask, load_puncta, save_mask, save_puncta, save_transform
from .principal_curve import compute_centroids, fit_principal_curve, sort_anchors
from .transform import apply_transform_to_points, transform_mask, unroll_clone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unroll 3D tissue clones along their principal curve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  unroll --mask-dir masks/ --puncta-dir puncta/pixel/ --output-dir output/

  # Custom parameters
  unroll --mask-dir masks/ --puncta-dir puncta/pixel/ --output-dir output/ \\
         --n-anchors 20 --plane yz --padding 100
        """,
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Directory containing 3D mask TIFF files",
    )
    parser.add_argument(
        "--puncta-dir",
        type=str,
        required=True,
        help="Directory containing puncta CSV files (x,y,z columns)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--n-anchors",
        type=int,
        default=30,
        help="Number of anchor points on the principal curve (default: 30)",
    )
    parser.add_argument(
        "--plane",
        type=str,
        default="zx",
        choices=["xy", "yz", "zx"],
        help="Unrolling plane (default: zx)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=50,
        help="Padding around unrolled mask (default: 50)",
    )
    parser.add_argument(
        "--epg-mu",
        type=float,
        default=1.0,
        help="ElPiGraph stretching penalty (default: 1.0)",
    )
    parser.add_argument(
        "--epg-lambda",
        type=float,
        default=0.01,
        help="ElPiGraph bending penalty (default: 0.01)",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for mask files (default: *.tif)",
    )
    return parser.parse_args(argv)


def extract_brain_clone_id(filename: str) -> tuple[str, str]:
    """Extract brain ID and clone ID from a mask filename.

    Expected pattern: *_brain{ID}_*_clone{N}_*
    """
    brain_match = re.search(r"brain(\d+)", filename)
    clone_match = re.search(r"clone(\d+)", filename)

    brain_id = brain_match.group(1) if brain_match else None
    clone_id = clone_match.group(1) if clone_match else "1"

    return brain_id, clone_id


def find_matching_puncta(puncta_dir: Path, brain_id: str) -> list[Path]:
    """Find puncta CSV files matching a brain ID."""
    pattern = f"*brain{brain_id}_*"
    return sorted(puncta_dir.glob(pattern))


def plot_qc(
    centroids: dict[int, np.ndarray],
    transformed_centroids: dict[int, np.ndarray],
    assignments: dict[int, int],
    anchor_positions: np.ndarray,
    anchors_ordered: np.ndarray,
    curve_trans: np.ndarray,
    mask: np.ndarray,
    transformed_mask: np.ndarray,
    output_offset: np.ndarray,
    save_path: Path,
):
    """Generate QC plot: original vs transformed tissue.

    Row 1: Cell centroids colored by anchor assignment + principal curve.
    Row 2: Actual mask voxels (subsampled) colored by label + principal curve.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 14))

    # Collect centroid data
    cell_ids = list(centroids.keys())
    orig_coords = np.array([centroids[cid] for cid in cell_ids])
    trans_coords = np.array([transformed_centroids[cid] for cid in cell_ids])
    anchor_ids = np.array([assignments[cid] for cid in cell_ids])

    ordered_anchor_orig = anchor_positions[anchors_ordered]

    # --- Row 1, Left: Original centroids ---
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(
        orig_coords[:, 2], orig_coords[:, 1], orig_coords[:, 0],
        c=anchor_ids, cmap="tab20", s=5, alpha=0.6,
    )
    ax1.plot(
        ordered_anchor_orig[:, 2], ordered_anchor_orig[:, 1], ordered_anchor_orig[:, 0],
        "k-o", markersize=4, linewidth=2, label="Principal curve",
    )
    ax1.scatter(
        [ordered_anchor_orig[0, 2]], [ordered_anchor_orig[0, 1]], [ordered_anchor_orig[0, 0]],
        c="red", s=100, marker="*", zorder=10, label="Start",
    )
    ax1.scatter(
        [ordered_anchor_orig[-1, 2]], [ordered_anchor_orig[-1, 1]], [ordered_anchor_orig[-1, 0]],
        c="blue", s=100, marker="*", zorder=10, label="End",
    )
    ax1.set_title("Original centroids")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(fontsize=7)

    # --- Row 1, Right: Transformed centroids ---
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.scatter(
        trans_coords[:, 2], trans_coords[:, 1], trans_coords[:, 0],
        c=anchor_ids, cmap="tab20", s=5, alpha=0.6,
    )
    ax2.plot(
        curve_trans[:, 2], curve_trans[:, 1], curve_trans[:, 0],
        "k-o", markersize=4, linewidth=2, label="Unrolled curve",
    )
    ax2.scatter(
        [curve_trans[0, 2]], [curve_trans[0, 1]], [curve_trans[0, 0]],
        c="red", s=100, marker="*", zorder=10, label="Start",
    )
    ax2.scatter(
        [curve_trans[-1, 2]], [curve_trans[-1, 1]], [curve_trans[-1, 0]],
        c="blue", s=100, marker="*", zorder=10, label="End",
    )
    ax2.set_title("Unrolled centroids")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend(fontsize=7)

    # --- Row 2: Mask voxels (subsampled for performance) ---
    max_voxels = 50000  # subsample to keep plot fast

    # Original mask voxels
    fg_orig = np.argwhere(mask > 0)
    labels_orig = mask[fg_orig[:, 0], fg_orig[:, 1], fg_orig[:, 2]]
    if len(fg_orig) > max_voxels:
        idx = np.random.default_rng(42).choice(len(fg_orig), max_voxels, replace=False)
        fg_orig = fg_orig[idx]
        labels_orig = labels_orig[idx]

    ax3 = fig.add_subplot(223, projection="3d")
    ax3.scatter(
        fg_orig[:, 2], fg_orig[:, 1], fg_orig[:, 0],
        c=labels_orig, cmap="tab20", s=0.5, alpha=0.3,
    )
    ax3.plot(
        ordered_anchor_orig[:, 2], ordered_anchor_orig[:, 1], ordered_anchor_orig[:, 0],
        "k-o", markersize=4, linewidth=2, label="Principal curve",
    )
    ax3.scatter(
        [ordered_anchor_orig[0, 2]], [ordered_anchor_orig[0, 1]], [ordered_anchor_orig[0, 0]],
        c="red", s=100, marker="*", zorder=10, label="Start",
    )
    ax3.scatter(
        [ordered_anchor_orig[-1, 2]], [ordered_anchor_orig[-1, 1]], [ordered_anchor_orig[-1, 0]],
        c="blue", s=100, marker="*", zorder=10, label="End",
    )
    ax3.set_title("Original mask voxels")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.legend(fontsize=7)

    # Transformed mask voxels
    fg_trans = np.argwhere(transformed_mask > 0)
    labels_trans = transformed_mask[fg_trans[:, 0], fg_trans[:, 1], fg_trans[:, 2]]
    # Convert from output array indices back to transformed coordinates
    fg_trans_coords = fg_trans.astype(float) + output_offset
    if len(fg_trans_coords) > max_voxels:
        idx = np.random.default_rng(42).choice(len(fg_trans_coords), max_voxels, replace=False)
        fg_trans_coords = fg_trans_coords[idx]
        labels_trans = labels_trans[idx]

    ax4 = fig.add_subplot(224, projection="3d")
    ax4.scatter(
        fg_trans_coords[:, 2], fg_trans_coords[:, 1], fg_trans_coords[:, 0],
        c=labels_trans, cmap="tab20", s=0.5, alpha=0.3,
    )
    ax4.plot(
        curve_trans[:, 2], curve_trans[:, 1], curve_trans[:, 0],
        "k-o", markersize=4, linewidth=2, label="Unrolled curve",
    )
    ax4.scatter(
        [curve_trans[0, 2]], [curve_trans[0, 1]], [curve_trans[0, 0]],
        c="red", s=100, marker="*", zorder=10, label="Start",
    )
    ax4.scatter(
        [curve_trans[-1, 2]], [curve_trans[-1, 1]], [curve_trans[-1, 0]],
        c="blue", s=100, marker="*", zorder=10, label="End",
    )
    ax4.set_title("Unrolled mask voxels")
    ax4.set_xlabel("X"); ax4.set_ylabel("Y"); ax4.set_zlabel("Z")
    ax4.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_one_mask(
    mask_path: Path,
    puncta_dir: Path,
    output_dir: Path,
    n_anchors: int,
    plane: str,
    padding: int,
    epg_mu: float,
    epg_lambda: float,
):
    """Process a single mask file: unroll mask and transform matching puncta."""
    brain_id, clone_id = extract_brain_clone_id(mask_path.name)
    if brain_id is None:
        logger.warning("Could not extract brain ID from %s, skipping", mask_path.name)
        return

    sample_name = f"brain{brain_id}_clone{clone_id}"
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %s", sample_name)

    # 1. Load mask and compute centroids
    logger.info("  Loading mask: %s", mask_path.name)
    mask = load_mask(mask_path)
    centroids = compute_centroids(mask)
    logger.info("  Found %d cells", len(centroids))

    # 2. Detect endpoints
    start_cell, end_cell = detect_endpoints(mask, centroids)
    logger.info(
        "  Endpoints: start=cell %d (neuroblast), end=cell %d", start_cell, end_cell
    )

    # 3. Fit principal curve
    logger.info("  Fitting elastic principal curve (%d anchors)", n_anchors)
    anchor_positions, edges, node_degree, assignments = fit_principal_curve(
        centroids, n_anchors=n_anchors, epg_mu=epg_mu, epg_lambda=epg_lambda
    )

    # 4. Sort anchors from start to end
    start_anchor, end_anchor = find_endpoint_anchors(
        anchor_positions, node_degree, centroids[start_cell], centroids[end_cell]
    )
    anchors_ordered = sort_anchors(
        edges, node_degree, anchor_positions, start_anchor, end_anchor
    )
    # Force first anchor to neuroblast centroid
    anchor_positions[anchors_ordered[0]] = centroids[start_cell]
    assignments[start_cell] = int(anchors_ordered[0])
    logger.info(
        "  Anchors sorted: start_anchor=%d, end_anchor=%d", start_anchor, end_anchor
    )

    # 5. Unroll
    logger.info("  Unrolling (plane=%s)", plane)
    transformed_centroids, curve_trans, transform_params = unroll_clone(
        centroids, anchors_ordered, anchor_positions, assignments,
        start_cell_id=start_cell, plane=plane,
    )

    # 6. Transform mask
    logger.info("  Transforming mask")
    transformed_mask, output_offset = transform_mask(
        mask, transform_params, padding=padding
    )
    save_mask(transformed_mask, sample_dir / "unrolled_mask.tif")
    logger.info("  Saved unrolled mask: %s", transformed_mask.shape)

    # Store offset in transform params
    transform_params["output_offset"] = output_offset.tolist()
    save_transform(transform_params, sample_dir / "transform.json")

    # 7. QC plot
    logger.info("  Generating QC plot")
    plot_qc(
        centroids, transformed_centroids, assignments,
        anchor_positions, anchors_ordered, curve_trans,
        mask, transformed_mask, output_offset,
        sample_dir / "qc_plot.png",
    )

    # 8. Transform matching puncta
    puncta_files = find_matching_puncta(puncta_dir, brain_id)
    if not puncta_files:
        logger.warning("  No puncta files found for brain%s", brain_id)
        return

    puncta_out_dir = sample_dir / "puncta"
    puncta_out_dir.mkdir(exist_ok=True)

    logger.info("  Transforming %d puncta files", len(puncta_files))
    for pf in puncta_files:
        df = load_puncta(pf)

        # Convert from (x, y, z) to (z, y, x) for mask operations
        points_zyx = df[["z", "y", "x"]].values

        transformed_pts, cell_ids = apply_transform_to_points(
            points_zyx, mask, transform_params
        )

        # Drop puncta outside any cell
        valid = cell_ids > 0
        if valid.sum() == 0:
            logger.warning("    %s: no puncta inside cells, skipping", pf.name)
            continue

        df_out = df[valid].copy()
        # Write transformed coords back as (x, y, z)
        df_out["x"] = transformed_pts[valid, 2]
        df_out["y"] = transformed_pts[valid, 1]
        df_out["z"] = transformed_pts[valid, 0]
        df_out["cell_id"] = cell_ids[valid]

        save_puncta(df_out, puncta_out_dir / pf.name)

    logger.info("  Done: %s", sample_name)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    mask_dir = Path(args.mask_dir)
    puncta_dir = Path(args.puncta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(mask_dir.glob(args.mask_pattern))
    if not mask_files:
        logger.error("No mask files found in %s with pattern %s", mask_dir, args.mask_pattern)
        sys.exit(1)

    logger.info("Found %d mask files", len(mask_files))

    for mask_path in mask_files:
        process_one_mask(
            mask_path=mask_path,
            puncta_dir=puncta_dir,
            output_dir=output_dir,
            n_anchors=args.n_anchors,
            plane=args.plane,
            padding=args.padding,
            epg_mu=args.epg_mu,
            epg_lambda=args.epg_lambda,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
