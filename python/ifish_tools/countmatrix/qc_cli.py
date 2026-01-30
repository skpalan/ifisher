"""CLI for count matrix QC plot generation."""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import tifffile

from .qc import plot_clone_qc, summarize_clone


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate QC plots for count matrices (.h5ad files).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  count-qc --matrix-dir /path/to/h5ad --puncta-dir /path/to/puncta --mask-dir /path/to/masks
  count-qc --matrix-dir /path/to/h5ad --puncta-dir /path/to/puncta --mask-dir /path/to/masks --output-dir /custom/qc/dir
        """,
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Directory containing .h5ad count matrix files.",
    )
    parser.add_argument(
        "--puncta-dir",
        type=Path,
        required=True,
        help="Directory containing registered puncta CSV files.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing clone mask TIFF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        help="Output directory for QC plots (default: {matrix-dir}/qc).",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="*_cp_masks.tif",
        help="Glob pattern for mask files (default: *_cp_masks.tif).",
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

    matrix_dir = args.matrix_dir
    puncta_dir = args.puncta_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir if args.output_dir else matrix_dir / "qc"

    # Validate directories
    if not matrix_dir.is_dir():
        log.error(f"Matrix directory not found: {matrix_dir}")
        sys.exit(1)
    if not puncta_dir.is_dir():
        log.error(f"Puncta directory not found: {puncta_dir}")
        sys.exit(1)
    if not mask_dir.is_dir():
        log.error(f"Mask directory not found: {mask_dir}")
        sys.exit(1)

    # Find all h5ad files
    h5ad_files = sorted(matrix_dir.glob("brain*.h5ad"))
    if not h5ad_files:
        log.error(f"No .h5ad files found in {matrix_dir}")
        sys.exit(1)

    log.info(f"Found {len(h5ad_files)} clone matrices")
    log.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table data
    summaries = []

    for h5ad_path in h5ad_files:
        clone_name = h5ad_path.stem  # e.g., brain08_clone1
        log.info(f"\nProcessing {clone_name}...")

        # Load AnnData
        adata = ad.read_h5ad(h5ad_path)
        brain_id = adata.obs["brain_id"].iloc[0]
        clone_id = adata.obs["clone_id"].iloc[0]

        # Generate summary
        summary = summarize_clone(adata, clone_name)
        summaries.append(summary)

        # Find corresponding mask
        # Build pattern carefully to avoid ** in glob
        mask_pattern = args.mask_pattern.lstrip('*')  # Remove leading * if present
        mask_candidates = list(mask_dir.glob(f"*brain{brain_id}*clone{clone_id}{mask_pattern}"))
        if not mask_candidates:
            # Try without mask_pattern suffix if no match
            mask_candidates = list(mask_dir.glob(f"*brain{brain_id}*clone{clone_id}*.tif"))
        
        if not mask_candidates:
            log.warning(f"  No mask found for brain {brain_id} clone {clone_id}, skipping")
            continue
        
        mask_path = mask_candidates[0]
        if len(mask_candidates) > 1:
            log.warning(f"  Multiple masks found, using: {mask_path.name}")

        # Find corresponding puncta CSVs (already in cropped brain space)
        puncta_paths = sorted(puncta_dir.glob(f"*_brain{brain_id}_*gene-*_regis.csv"))
        if not puncta_paths:
            # Try without _regis suffix
            puncta_paths = sorted(puncta_dir.glob(f"*_brain{brain_id}_*gene-*.csv"))
        
        if not puncta_paths:
            log.warning(f"  No puncta CSVs found for brain {brain_id}, skipping")
            continue
        
        log.info(f"  Found {len(puncta_paths)} puncta CSVs")

        # Auto-detect z_scale from first puncta file
        z_scale = 1
        if puncta_paths:
            try:
                sample_df = pd.read_csv(puncta_paths[0])
                mask_tmp = tifffile.imread(mask_path)
                z_scale = round(mask_tmp.shape[0] / (sample_df["z"].max() + 1))
            except Exception as e:
                log.warning(f"  Could not auto-detect z_scale: {e}, using z_scale=1")

        log.info(f"  Z-scale auto-detection: z_scale={z_scale}")

        # Generate QC plot
        output_path = output_dir / f"{clone_name}_qc.png"
        try:
            plot_clone_qc(
                adata, clone_name, summary, mask_path,
                puncta_paths, z_scale,
                output_path
            )
        except Exception as e:
            log.error(f"  Failed to generate QC plot: {e}")
            continue

    # Save summary table
    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"\nSaved summary table: {summary_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Flag anomalies
    anomalies = []
    for _, row in summary_df.iterrows():
        if row["total_counts"] == 0:
            anomalies.append(f"{row['clone']}: ZERO total counts!")
        elif row["median_counts_per_cell"] < 5:
            anomalies.append(f"{row['clone']}: Very low median counts ({row['median_counts_per_cell']:.1f})")
        if row["zero_count_cells"] > 0:
            anomalies.append(f"{row['clone']}: {row['zero_count_cells']} zero-count cells")

    if anomalies:
        print("\n⚠️  ANOMALIES DETECTED:")
        for a in anomalies:
            print(f"  - {a}")
    else:
        print("\n✓ No major anomalies detected.")

    log.info(f"\nQC complete. All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
