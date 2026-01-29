"""CLI for building gene x cell count matrices."""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .core import process_clone


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build gene x cell count matrices from 3D masks and puncta CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  count-matrix --mask-dir /path/to/masks --puncta-dir /path/to/puncta --output-dir /path/to/output
  count-matrix --mask-dir /path/to/masks --puncta-dir /path/to/puncta --output-dir /path/to/output --workers 4
        """,
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing clone mask TIFF files.",
    )
    parser.add_argument(
        "--puncta-dir",
        type=Path,
        required=True,
        help="Directory containing puncta CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .h5ad files.",
    )
    parser.add_argument(
        "--bbox-dir",
        type=Path,
        required=False,
        help="Directory containing bbox_ref.mat files (e.g., regis_downsamp_1121/). "
             "If provided, applies coordinate correction (crop offset + z-scaling).",
    )
    parser.add_argument(
        "--save-puncta",
        type=Path,
        required=False,
        help="Directory to save cropped puncta CSVs (transformed to mask coordinates). "
             "Requires --bbox-dir to be set.",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="*_cp_masks.tif",
        help="Glob pattern for mask files (default: *_cp_masks.tif).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    mask_dir = args.mask_dir
    puncta_dir = args.puncta_dir
    output_dir = args.output_dir
    bbox_dir = args.bbox_dir
    save_puncta_dir = args.save_puncta

    if not mask_dir.is_dir():
        log.error(f"Mask directory not found: {mask_dir}")
        sys.exit(1)
    if not puncta_dir.is_dir():
        log.error(f"Puncta directory not found: {puncta_dir}")
        sys.exit(1)
    if bbox_dir is not None and not bbox_dir.is_dir():
        log.error(f"Bbox directory not found: {bbox_dir}")
        sys.exit(1)
    if save_puncta_dir is not None and bbox_dir is None:
        log.error("--save-puncta requires --bbox-dir to be set")
        sys.exit(1)

    mask_paths = sorted(mask_dir.glob(args.mask_pattern))
    if not mask_paths:
        log.error(f"No mask files found matching '{args.mask_pattern}' in {mask_dir}")
        sys.exit(1)

    log.info(f"Found {len(mask_paths)} clone masks")
    log.info(f"Output directory: {output_dir}")
    if bbox_dir:
        log.info(f"Bbox directory: {bbox_dir} (coordinate correction enabled)")
    else:
        log.warning("No bbox directory provided - using raw coordinates (may be inaccurate)")
    if save_puncta_dir:
        log.info(f"Cropped puncta will be saved to: {save_puncta_dir}")

    if args.workers > 1 and len(mask_paths) > 1:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_clone, mp, puncta_dir, output_dir, bbox_dir, save_puncta_dir): mp
                for mp in mask_paths
            }
            for future in as_completed(futures):
                mp = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    log.error(f"Failed processing {mp.name}: {e}")
    else:
        results = []
        for mp in mask_paths:
            try:
                result = process_clone(mp, puncta_dir, output_dir, bbox_dir, save_puncta_dir)
                if result:
                    results.append(result)
            except Exception as e:
                log.error(f"Failed processing {mp.name}: {e}")

    log.info(f"Done. {len(results)}/{len(mask_paths)} clones processed successfully.")


if __name__ == "__main__":
    main()
