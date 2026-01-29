"""
RS-FISH Batch Runner for iFISH Data

Batch process HDF5 files with rs-fish for spot detection.
Handles brain-to-timepoint mapping, threshold lookup, and parallel execution.

Usage:
    # As a module
    from ifish_tools.rsfish_runner import run_rsfish_batch, find_files_to_process
    
    # As CLI
    python -m ifish_tools.rsfish_runner \\
        --data-dir /scratch/Gel_1029/data/h5_16bit \\
        --metadata /path/to/nd2_metadata.csv \\
        --thresholds /path/to/threshold.csv \\
        --output-dir /scratch/Gel_1029/puncta_0121 \\
        --sigma 1.3 \\
        --workers 20

Features:
    - Automatic brain-to-timepoint mapping from metadata CSV
    - Threshold lookup per round/channel/timepoint
    - Parallel execution with configurable worker count
    - Dry-run mode for testing
    - Progress tracking with tqdm
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Default rs-fish executable path (use 'rs-fish' from PATH, provided by rs-fish module)
DEFAULT_RSFISH_CMD = "rs-fish"

# Channel name mapping from h5 filenames to threshold CSV column names
CHANNEL_NAME_MAP = {
    "mCherry Nar": "mCherry_Nar",
    "Cy3 Nar": "Cy3_Nar",
    "Cy5": "Cy5",
    "DAPI": "DAPI",
    "FITC-GFP": "FITC",
}

# Reverse mapping for output
CHANNEL_NAME_REVERSE = {v: k for k, v in CHANNEL_NAME_MAP.items()}


@dataclass
class RSFishJob:
    """Represents a single rs-fish job to execute."""
    h5_path: str
    output_csv: str
    threshold: float
    gene_name: str
    brain: str
    round_num: int
    channel: str
    sigma: float = 1.3
    anisotropy: float = 0.49
    threads: int = 4


def load_brain_timepoint_mapping(metadata_csv: str) -> dict[str, str]:
    """
    Load brain-to-timepoint mapping from nd2_metadata.csv.
    
    The CSV has columns: New file name, Original filename, Relative path
    Original filename contains patterns like "lAL_72_10" or "lAL_96_1"
    
    Parameters
    ----------
    metadata_csv : str
        Path to nd2_metadata.csv
        
    Returns
    -------
    dict[str, str]
        Mapping of brain number (e.g., '08') to timepoint ('72h' or '96h')
    """
    brain_timepoints = {}
    
    with open(metadata_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_name = row.get('New file name', '')
            original = row.get('Original filename', '')
            
            # Extract brain number from "08.nd2" format
            brain_match = re.match(r'^(\d+)\.nd2$', new_name)
            if not brain_match:
                continue
            brain_num = brain_match.group(1)
            
            # Extract timepoint from original filename
            # Pattern: XX_lAL_72_YY or XX_lAL_96_YY
            if '_lAL_72_' in original or 'lAL_72_' in original:
                brain_timepoints[brain_num] = '72h'
            elif '_lAL_96_' in original or 'lAL_96_' in original:
                brain_timepoints[brain_num] = '96h'
            elif '_lAL_48_' in original or 'lAL_48_' in original:
                # Skip 48h brains (e.g., brain17)
                brain_timepoints[brain_num] = '48h'
    
    return brain_timepoints


def load_threshold_table(threshold_csv: str) -> dict[tuple, tuple]:
    """
    Load threshold table from CSV.
    
    The CSV has columns:
    - Cycle (round number)
    - Gene
    - Initiator
    - Imaging Channel
    - 96hr(brain_09) - threshold for 96h
    - 72hr(brain_15) - threshold for 72h
    
    Parameters
    ----------
    threshold_csv : str
        Path to threshold CSV file
        
    Returns
    -------
    dict[tuple, tuple]
        Mapping of (round, imaging_channel, timepoint) -> (threshold, gene_name)
        Returns None for threshold if missing or invalid
    """
    thresholds = {}
    
    with open(threshold_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycle = int(row.get('Cycle', -1))
            except (ValueError, TypeError):
                continue
            
            gene = row.get('Gene', '').strip()
            channel = row.get('Imaging Channel', '').strip()
            
            if not gene or not channel:
                continue
            
            # Skip DAPI and FITC channels
            if channel in ('DAPI', 'FITC'):
                continue
            
            # Parse thresholds for each timepoint
            for col_name, timepoint in [('96hr(brain_09)', '96h'), ('72hr(brain_15)', '72h')]:
                val = row.get(col_name, '').strip()
                
                # Skip empty, invalid, or special values
                if not val or val.lower() in ('', 'small file?'):
                    threshold = None
                else:
                    try:
                        threshold = float(val)
                    except ValueError:
                        threshold = None
                
                key = (cycle, channel, timepoint)
                thresholds[key] = (threshold, gene)
    
    return thresholds


def parse_h5_filename(filename: str) -> Optional[dict]:
    """
    Parse h5 filename to extract metadata.
    
    Expected format: Gel20251024_round01_brain08_channel-mCherry Nar.h5
    
    Parameters
    ----------
    filename : str
        H5 filename (basename only)
        
    Returns
    -------
    dict or None
        Dictionary with keys: gel_date, round, brain, channel
        Returns None if parsing fails
    """
    # Pattern: Gel{DATE}_round{NN}_brain{NN}_channel-{CHANNEL}.h5
    pattern = r'^(Gel\d+)_round(\d+)_brain(\d+)_channel-(.+)\.h5$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    return {
        'gel_date': match.group(1),
        'round': int(match.group(2)),
        'brain': match.group(3),
        'channel': match.group(4),
    }


def find_files_to_process(
    data_dir: str,
    brain_timepoints: dict[str, str],
    thresholds: dict[tuple, tuple],
    output_dir: str,
    exclude_rounds: list[int] = None,
    exclude_channels: list[str] = None,
    force: bool = False,
) -> list[RSFishJob]:
    """
    Find all H5 files to process and build job list.
    
    Parameters
    ----------
    data_dir : str
        Directory containing H5 files
    brain_timepoints : dict
        Mapping of brain number to timepoint
    thresholds : dict
        Threshold lookup table
    output_dir : str
        Output directory (used to check for existing files)
    exclude_rounds : list[int], optional
        Round numbers to exclude (default: [0, 10] for round00 and round10)
    exclude_channels : list[str], optional
        Channel names to exclude (default: ['DAPI', 'FITC-GFP'])
    force : bool, optional
        If True, overwrite existing output files. Default is False.
        
    Returns
    -------
    list[RSFishJob]
        List of jobs to execute
    """
    if exclude_rounds is None:
        exclude_rounds = [0, 10]  # Skip round00 (reference) and round10 (not in metadata)
    if exclude_channels is None:
        exclude_channels = ['DAPI', 'FITC-GFP']
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    jobs = []
    skipped = {'no_timepoint': [], 'excluded_round': [], 'excluded_channel': [], 
               'no_threshold': [], 'parse_error': [], 'already_exists': []}
    
    for h5_file in sorted(data_path.glob('*.h5')):
        filename = h5_file.name
        parsed = parse_h5_filename(filename)
        
        if parsed is None:
            skipped['parse_error'].append(filename)
            continue
        
        # Check exclusions
        if parsed['round'] in exclude_rounds:
            skipped['excluded_round'].append(filename)
            continue
        
        if parsed['channel'] in exclude_channels:
            skipped['excluded_channel'].append(filename)
            continue
        
        # Get timepoint for this brain
        timepoint = brain_timepoints.get(parsed['brain'])
        if timepoint is None or timepoint == '48h':
            skipped['no_timepoint'].append(filename)
            continue
        
        # Map channel name to CSV column name
        channel_csv = CHANNEL_NAME_MAP.get(parsed['channel'], parsed['channel'])
        
        # Look up threshold
        key = (parsed['round'], channel_csv, timepoint)
        threshold_info = thresholds.get(key)
        
        if threshold_info is None or threshold_info[0] is None:
            skipped['no_threshold'].append(filename)
            continue
        
        threshold, gene_name = threshold_info
        
        # Build output filename: Gel20251024_brain08_round01_gene-dati.csv
        output_name = f"{parsed['gel_date']}_brain{parsed['brain']}_round{parsed['round']:02d}_gene-{gene_name}.csv"
        
        # Check if output already exists
        output_file = output_path / output_name
        if output_file.exists() and not force:
            skipped['already_exists'].append(filename)
            continue
        
        jobs.append(RSFishJob(
            h5_path=str(h5_file),
            output_csv=output_name,
            threshold=threshold,
            gene_name=gene_name,
            brain=parsed['brain'],
            round_num=parsed['round'],
            channel=parsed['channel'],
        ))
    
    # Print summary of skipped files
    print(f"\nFile discovery summary:")
    print(f"  Found {len(jobs)} files to process")
    print(f"  Skipped (excluded round): {len(skipped['excluded_round'])}")
    print(f"  Skipped (excluded channel): {len(skipped['excluded_channel'])}")
    print(f"  Skipped (no timepoint/48h): {len(skipped['no_timepoint'])}")
    print(f"  Skipped (no threshold): {len(skipped['no_threshold'])}")
    print(f"  Skipped (already exists): {len(skipped['already_exists'])}")
    print(f"  Skipped (parse error): {len(skipped['parse_error'])}")
    
    if skipped['no_threshold']:
        print(f"\n  Files skipped due to missing thresholds:")
        for f in skipped['no_threshold'][:10]:
            print(f"    - {f}")
        if len(skipped['no_threshold']) > 10:
            print(f"    ... and {len(skipped['no_threshold']) - 10} more")
    
    return jobs


def build_rsfish_command(
    job: RSFishJob,
    output_dir: str,
    rsfish_cmd: str = DEFAULT_RSFISH_CMD,
) -> list[str]:
    """
    Build rs-fish command as a list of arguments.
    
    Parameters
    ----------
    job : RSFishJob
        Job specification
    output_dir : str
        Output directory for CSV files
    rsfish_cmd : str
        Path to rs-fish executable
        
    Returns
    -------
    list[str]
        Command as list of arguments for subprocess
    """
    output_path = Path(output_dir) / job.output_csv
    
    cmd = [
        rsfish_cmd,
        '-i', job.h5_path,
        '-o', str(output_path),
        '-s', str(job.sigma),
        '-t', str(job.threshold),
        '-a', str(job.anisotropy),
        '--threads', str(job.threads),
    ]
    
    return cmd


def run_single_rsfish(args: tuple) -> dict:
    """
    Worker function to run a single rs-fish job.
    
    Parameters
    ----------
    args : tuple
        (job, output_dir, rsfish_cmd, dry_run)
        
    Returns
    -------
    dict
        Result with keys: success, job, output, error
    """
    job, output_dir, rsfish_cmd, dry_run = args
    
    cmd = build_rsfish_command(job, output_dir, rsfish_cmd)
    
    result = {
        'success': False,
        'job': job,
        'command': ' '.join(cmd),
        'output': '',
        'error': '',
    }
    
    if dry_run:
        result['success'] = True
        result['output'] = '[DRY RUN] Would execute command'
        return result
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per job
        )
        
        result['output'] = proc.stdout
        result['error'] = proc.stderr
        result['success'] = proc.returncode == 0
        
        if proc.returncode != 0:
            result['error'] = f"Return code: {proc.returncode}\n{proc.stderr}"
            
    except subprocess.TimeoutExpired:
        result['error'] = "Job timed out after 1 hour"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_rsfish_batch(
    jobs: list[RSFishJob],
    output_dir: str,
    max_workers: int = 20,
    rsfish_cmd: str = DEFAULT_RSFISH_CMD,
    sigma: float = 1.3,
    anisotropy: float = 0.49,
    threads_per_job: int = 4,
    dry_run: bool = False,
) -> dict:
    """
    Execute all rs-fish jobs in parallel.
    
    Parameters
    ----------
    jobs : list[RSFishJob]
        List of jobs to execute
    output_dir : str
        Output directory for CSV files
    max_workers : int
        Maximum number of parallel workers
    rsfish_cmd : str
        Path to rs-fish executable
    sigma : float
        Sigma parameter for rs-fish
    anisotropy : float
        Anisotropy parameter for rs-fish
    threads_per_job : int
        Number of threads per rs-fish job
    dry_run : bool
        If True, only print commands without executing
        
    Returns
    -------
    dict
        Summary with counts of successful, failed jobs and error details
    """
    # Update job parameters
    for job in jobs:
        job.sigma = sigma
        job.anisotropy = anisotropy
        job.threads = threads_per_job
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"RS-FISH Batch Runner")
    print(f"{'='*60}")
    print(f"Jobs to process: {len(jobs)}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {max_workers}")
    print(f"Threads per job: {threads_per_job}")
    print(f"Sigma: {sigma}")
    print(f"Anisotropy: {anisotropy}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")
    
    # Prepare task arguments
    task_args = [(job, output_dir, rsfish_cmd, dry_run) for job in jobs]
    
    results = {
        'successful': [],
        'failed': [],
        'total': len(jobs),
    }
    
    if dry_run:
        print("DRY RUN - Commands that would be executed:\n")
        for i, (job, _, _, _) in enumerate(task_args[:5]):
            cmd = build_rsfish_command(job, output_dir, rsfish_cmd)
            print(f"[{i+1}] {' '.join(cmd)}\n")
        if len(task_args) > 5:
            print(f"... and {len(task_args) - 5} more jobs\n")
        
        results['successful'] = jobs
        return results
    
    # Execute jobs in parallel
    if HAS_TQDM:
        pbar = tqdm(total=len(jobs), desc="Processing", unit="file")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_rsfish, args): args[0] for args in task_args}
        
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if result['success']:
                    results['successful'].append(job)
                else:
                    results['failed'].append({
                        'job': job,
                        'error': result['error'],
                    })
                    print(f"\nFailed: {job.output_csv}")
                    print(f"  Error: {result['error'][:200]}")
            except Exception as e:
                results['failed'].append({
                    'job': job,
                    'error': str(e),
                })
                print(f"\nException for {job.output_csv}: {e}")
            
            if HAS_TQDM:
                pbar.update(1)
    
    if HAS_TQDM:
        pbar.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"{'='*60}")
    
    if results['failed']:
        print("\nFailed jobs:")
        for item in results['failed'][:10]:
            print(f"  - {item['job'].output_csv}: {item['error'][:100]}")
        if len(results['failed']) > 10:
            print(f"  ... and {len(results['failed']) - 10} more")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Batch run rs-fish on HDF5 files with automatic threshold lookup.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard batch processing
  python -m ifish_tools.rsfish_runner \\
      --data-dir /scratch/Gel_1029/data/h5_16bit \\
      --metadata /lsi/groups/tleelab/Hui-Min/lAL_72-96/nd2_metadata.csv \\
      --thresholds /scratch/Gel_1029/puncta_0121/lAL_72_96_threshold_Jan2026.csv \\
      --output-dir /scratch/Gel_1029/puncta_0121 \\
      --sigma 1.3 \\
      --workers 20

  # Dry run to see what would be executed
  python -m ifish_tools.rsfish_runner ... --dry-run
        """
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing H5 files')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to nd2_metadata.csv for brain-timepoint mapping')
    parser.add_argument('--thresholds', type=str, required=True,
                        help='Path to threshold CSV file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for CSV files')
    
    parser.add_argument('--sigma', type=float, default=1.3,
                        help='Sigma for Difference-of-Gaussian (default: 1.3)')
    parser.add_argument('--anisotropy', type=float, default=0.49,
                        help='Anisotropy factor (default: 0.49)')
    parser.add_argument('--workers', type=int, default=20,
                        help='Number of parallel workers (default: 20)')
    parser.add_argument('--threads-per-job', type=int, default=4,
                        help='Threads per rs-fish job (default: 4)')
    
    parser.add_argument('--rsfish-cmd', type=str, default=DEFAULT_RSFISH_CMD,
                        help=f'Path to rs-fish executable (default: {DEFAULT_RSFISH_CMD})')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.data_dir).is_dir():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    if not Path(args.metadata).is_file():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)
    if not Path(args.thresholds).is_file():
        print(f"Error: Threshold file not found: {args.thresholds}")
        sys.exit(1)
    
    # Load mappings
    print("Loading brain-timepoint mapping...")
    brain_timepoints = load_brain_timepoint_mapping(args.metadata)
    print(f"  Found {len(brain_timepoints)} brain mappings")
    for brain, tp in sorted(brain_timepoints.items()):
        print(f"    brain{brain}: {tp}")
    
    print("\nLoading threshold table...")
    thresholds = load_threshold_table(args.thresholds)
    print(f"  Found {len(thresholds)} threshold entries")
    
    # Find files to process
    print("\nScanning for files to process...")
    jobs = find_files_to_process(
        args.data_dir, 
        brain_timepoints, 
        thresholds,
        args.output_dir,
        force=args.force,
    )
    
    if not jobs:
        print("\nNo files to process!")
        sys.exit(0)
    
    # Start timer
    start_time = time.time()
    
    # Run batch processing
    results = run_rsfish_batch(
        jobs=jobs,
        output_dir=args.output_dir,
        max_workers=args.workers,
        rsfish_cmd=args.rsfish_cmd,
        sigma=args.sigma,
        anisotropy=args.anisotropy,
        threads_per_job=args.threads_per_job,
        dry_run=args.dry_run,
    )
    
    # Print total time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    # Exit with error code if any jobs failed
    if results['failed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
