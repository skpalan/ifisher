"""CLI for clone mask extraction."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-clone segmentation masks with morphological closing."
    )
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--closing-radius", type=int, default=None,
                        help="Override closing radius (default: from config or 5).")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print plan without processing.")
    args = parser.parse_args()

    from .config import CloneExtractConfig
    from .core import run_pipeline

    config = CloneExtractConfig.from_yaml(args.config)
    if args.closing_radius is not None:
        config.closing_radius = args.closing_radius
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    if args.dry_run:
        print("=== Dry Run ===")
        print(f"Output dir: {config.output_dir}")
        print(f"Closing radius: {config.closing_radius}")
        print(f"Date tag: {config.date_tag}")
        for spec in config.brains:
            print(f"\nBrain: {spec.brain_name}")
            print(f"  Mask: {spec.mask_path}")
            print(f"  B-box shape (Z,Y,X): {spec.bbox.shape()}")
            for cname, cbox in spec.clones.items():
                print(f"  Clone {cname}: shape (Z,Y,X) = {cbox.shape()}")
        return

    paths = run_pipeline(config)
    print(f"\nDone. {len(paths)} files written.")


if __name__ == "__main__":
    main()
