"""
Command-line interface for u-segment3D pipeline.
"""

import argparse
import sys

from .pipeline import USeg3DPipeline
from .config import generate_config_template


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='u-segment3D pipeline for 3D cell segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with config file
  python -m ifish_tools.usegment3d --config config.yaml
  
  # Dry run to validate config
  python -m ifish_tools.usegment3d --config config.yaml --dry-run
  
  # Use specific GPUs
  python -m ifish_tools.usegment3d --config config.yaml --gpus 0,1
  
  # Process specific brains only
  python -m ifish_tools.usegment3d --config config.yaml --brains brain08,brain11
  
  # Generate template config
  python -m ifish_tools.usegment3d --generate-config template.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--generate-config',
        type=str,
        metavar='PATH',
        help='Generate template configuration file and exit'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config and show what would be processed without running'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=None,
        help='Skip brains with existing output files (overrides config)'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Process all brains even if outputs exist (overrides config)'
    )
    
    parser.add_argument(
        '--gpus',
        type=str,
        help='Comma-separated list of GPU IDs (e.g., "0,1")'
    )
    
    parser.add_argument(
        '--brains',
        type=str,
        help='Comma-separated list of brain names to process (e.g., "brain08,brain11")'
    )
    
    args = parser.parse_args()
    
    # Generate config template
    if args.generate_config:
        generate_config_template(args.generate_config)
        return 0
    
    # Check required arguments
    if not args.config:
        parser.error("--config is required (or use --generate-config)")
    
    # Parse optional arguments
    dry_run = args.dry_run
    
    skip_existing = None
    if args.skip_existing:
        skip_existing = True
    elif args.no_skip_existing:
        skip_existing = False
    
    gpus = None
    if args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(',')]
    
    brains = None
    if args.brains:
        brains = [b.strip() for b in args.brains.split(',')]
    
    # Run pipeline
    try:
        pipeline = USeg3DPipeline.from_yaml(args.config)
        results = pipeline.run(
            dry_run=dry_run,
            skip_existing=skip_existing,
            gpus=gpus,
            brains=brains
        )
        
        if results:
            # Check for errors
            errors = [name for name, r in results.items() if 'error' in r]
            if errors:
                print(f"\n⚠ {len(errors)} brain(s) failed: {', '.join(errors)}")
                return 1
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
