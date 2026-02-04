"""CLI for spatial gene expression 3D visualization."""

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 3D spatial expression heatmaps for unrolled tissue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - all genes, all clones
  plot-spatial --matrix-dir /path/to/h5ad --unroll-dir /path/to/unroll --output-dir /path/to/output

  # Specific genes
  plot-spatial --matrix-dir matrices/ --unroll-dir unroll/ --output-dir plots/ --genes ase Dl Hey

  # Parallel processing with 8 workers
  plot-spatial --matrix-dir matrices/ --unroll-dir unroll/ --output-dir plots/ --workers 8
        """,
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Directory containing .h5ad count matrix files.",
    )
    parser.add_argument(
        "--unroll-dir",
        type=Path,
        required=True,
        help="Directory containing unroll output (brain*/unrolled_mask.tif, transform.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--raw-mask-dir",
        type=Path,
        default=Path("/scratch/Gel_1029/matrix_01-29/module_segment_0130/masks_r12_0202"),
        help="Directory containing raw (non-unrolled) mask files.",
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="Specific genes to plot (default: all genes in AnnData).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdYlGn",
        help="Matplotlib colormap for expression (default: RdYlGn).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for PNG files (default: 300).",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=50000,
        help="Maximum voxels to plot per panel (default: 50000).",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.99,
        help="Minimum cell match ratio for h5ad↔transform.json (default: 0.99 = 99%%).",
    )
    parser.add_argument(
        "--min-mask-overlap",
        type=float,
        default=0.99,
        help="Minimum label overlap ratio for unrolled↔raw masks (default: 0.99 = 99%%).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point."""
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    # Validate input directories
    if not args.matrix_dir.exists():
        logger.error(f"Matrix directory not found: {args.matrix_dir}")
        sys.exit(1)

    if not args.unroll_dir.exists():
        logger.error(f"Unroll directory not found: {args.unroll_dir}")
        sys.exit(1)

    # Import here to avoid slow startup
    from .core import process_all_clones

    logger.info("Starting spatial expression plotting")
    logger.info(f"Matrix dir: {args.matrix_dir}")
    logger.info(f"Unroll dir: {args.unroll_dir}")
    logger.info(f"Raw mask dir: {args.raw_mask_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    if args.genes:
        logger.info(f"Genes: {args.genes}")
    else:
        logger.info("Genes: all")

    # Process all clones
    outputs = process_all_clones(
        matrix_dir=args.matrix_dir,
        unroll_dir=args.unroll_dir,
        raw_mask_dir=args.raw_mask_dir,
        output_dir=args.output_dir,
        genes=args.genes,
        cmap=args.cmap,
        dpi=args.dpi,
        max_voxels=args.max_voxels,
        workers=args.workers,
        min_match_ratio=args.min_match_ratio,
        min_mask_overlap=args.min_mask_overlap,
    )

    logger.info(f"Completed! Generated {len(outputs)} plots in {args.output_dir}")


if __name__ == "__main__":
    main()
