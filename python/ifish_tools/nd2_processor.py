"""
ND2 to H5/TIFF Processor

Fast conversion of Nikon ND2 files to HDF5 and TIFF formats with proper metadata.
This is a Python re-implementation of the MATLAB process_nd2_to_h5.m function,
using the fast nd2 library instead of Bio-Formats.

Features:
- Fast ND2 reading using the nd2 library
- Automatic averaging of replicate channels
- Support for h5_8bit, h5_16bit, tiff_8bit, tiff_16bit output formats
- Per-channel format specification
- Parallel channel processing (ThreadPoolExecutor)
- Parallel file processing (ProcessPoolExecutor)
- Memory-aware auto-configuration
- Skip-existing behavior by default
- Configurable compression (gzip-4, gzip-1, lzf, none)

Usage:
    # As a module
    from ifish_tools.nd2_processor import process_nd2_to_h5, process_nd2_folder
    
    process_nd2_folder(
        '/path/to/raw/nd2/folder',
        '/path/to/output/data',
        h5_16bit=['Cy3 Nar', 'Cy5', 'mCherry Nar'],
        h5_8bit=['DAPI', 'FITC-GFP'],
        workers=4,
        channel_workers=5,
    )
    
    # As CLI
    python -m ifish_tools.nd2_processor /path/to/nd2/folder /path/to/output \\
        --h5-16bit "Cy3 Nar" "Cy5" "mCherry Nar" \\
        --h5-8bit "DAPI" "FITC-GFP" \\
        --workers 4 --channel-workers 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import nd2
import numpy as np
from tifffile import TiffWriter

# Optional imports for progress bars and memory detection
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# Thread-safe print lock
_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


def _parse_memory_string(memory_str: str) -> float:
    """Parse memory string like '64G' or '256GB' to gigabytes."""
    memory_str = memory_str.strip().upper()
    match = re.match(r'^([\d.]+)\s*([KMGT]?)B?$', memory_str)
    if not match:
        raise ValueError(f"Invalid memory format: {memory_str}. Use format like '64G' or '256GB'.")
    
    value = float(match.group(1))
    unit = match.group(2)
    
    multipliers = {'': 1e-9, 'K': 1e-6, 'M': 1e-3, 'G': 1, 'T': 1000}
    return value * multipliers.get(unit, 1)


def _get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    if HAS_PSUTIL:
        return psutil.virtual_memory().available / 1e9
    else:
        # Fallback: assume 16GB available
        return 16.0


def _estimate_memory_per_file_gb(n_channels: int = 5) -> float:
    """
    Estimate memory usage per ND2 file in GB.
    
    Based on typical ND2 file:
    - Full array: ~16GB (361 x 11 x 1151 x 1815 x 2 bytes)
    - Per channel (float32): ~2.4GB (361 x 1815 x 1151 x 4 bytes)
    """
    nd2_size_gb = 16.0  # Typical ND2 file size
    channel_size_gb = 2.4  # Per channel as float32
    return nd2_size_gb + n_channels * channel_size_gb


def _auto_detect_settings(
    max_memory_gb: Optional[float] = None,
    n_channels: int = 5,
) -> tuple[int, int]:
    """
    Auto-detect optimal workers and channel_workers based on system resources.
    
    Parameters
    ----------
    max_memory_gb : float, optional
        Maximum memory to use in GB. If None, uses 80% of available.
    n_channels : int
        Expected number of channels per file.
        
    Returns
    -------
    tuple[int, int]
        (workers, channel_workers)
    """
    cpu_count = os.cpu_count() or 4
    available_memory_gb = _get_available_memory_gb()
    
    if max_memory_gb is None:
        max_memory_gb = available_memory_gb * 0.8
    
    memory_per_file_gb = _estimate_memory_per_file_gb(n_channels)
    
    # Calculate max workers based on memory
    max_workers_by_memory = max(1, int(max_memory_gb / memory_per_file_gb))
    
    # Calculate workers based on CPU (assume ~8 threads per file)
    max_workers_by_cpu = max(1, cpu_count // 8)
    
    workers = min(max_workers_by_memory, max_workers_by_cpu)
    
    # Channel workers: process all channels in parallel, but cap based on remaining CPU
    channel_workers = min(n_channels, max(1, cpu_count // (workers * 2)))
    
    return workers, channel_workers


def _parse_basename(nd2_path: Path) -> str:
    """
    Parse basename from ND2 file path following the naming convention.
    
    Input folder: Gel20260109_round00_GFP_lab_vnd_ind/01.nd2
    Output basename: Gel20260109_round00_brain01
    
    Also handles:
    - 6-digit dates (YYMMDD -> YYYYMMDD): Gel251115 -> Gel20251115
    - Complex filenames: "23_WT03 - Deconvolved..." -> brain23_WT03
    """
    parent_name = nd2_path.parent.name
    file_stem = nd2_path.stem
    
    match = re.match(r'(Gel)(\d+)_(round\d+)', parent_name)
    if match:
        gel_prefix = match.group(1)
        gel_date = match.group(2)
        round_id = match.group(3)
        
        # Convert 6-digit date (YYMMDD) to 8-digit (YYYYMMDD)
        if len(gel_date) == 6:
            # Assume 20xx for years, e.g., 251115 -> 20251115
            gel_date = '20' + gel_date
        
        gel_id = f"{gel_prefix}{gel_date}"
        
        # Parse brain number from filename
        # Handle formats like: "01", "02", "23_WT03 - Deconvolved..."
        if file_stem.isdigit():
            brain_num = f"brain{file_stem.zfill(2)}"
        else:
            # Extract leading number only (e.g., "23" from "23_WT03 - Deconvolved...")
            brain_match = re.match(r'^(\d+)', file_stem)
            if brain_match:
                num = brain_match.group(1)
                brain_num = f"brain{num.zfill(2)}"
            else:
                # Fallback: use cleaned file stem
                brain_num = f"brain_{re.sub(r'[^A-Za-z0-9_]', '_', file_stem)}"
        
        return f"{gel_id}_{round_id}_{brain_num}"
    else:
        return f"{parent_name}_{file_stem}"


def _normalize_to_8bit(data: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Normalize data to 8-bit using log-scale with percentile clipping.
    Matches MATLAB implementation.
    """
    data_float = data.astype(np.float32)
    data_float = np.maximum(data_float, 1e-10)
    
    data_log = np.log(data_float)
    
    log_min = np.percentile(data_log, 0.01)
    log_max = np.percentile(data_log, 99.99)
    
    data_log = np.clip(data_log, log_min, log_max)
    data_log = (data_log - log_min) / (log_max - log_min + 1e-10)
    
    data_8bit = (data_log * 255).astype(np.uint8)
    
    return data_8bit, float(log_max)


def _normalize_to_16bit(data: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Normalize data to 16-bit using linear scaling.
    Matches MATLAB behavior with rescale.
    """
    data_float = data.astype(np.float32)
    max_val = float(np.max(data_float))
    
    if max_val > 0:
        data_norm = data_float / max_val
    else:
        data_norm = data_float
    
    min_val = float(np.min(data_norm))
    max_norm = float(np.max(data_norm))
    if max_norm > min_val:
        data_norm = (data_norm - min_val) / (max_norm - min_val)
    
    data_16bit = (data_norm * 65535).astype(np.uint16)
    
    return data_16bit, max_val


def _get_compression_settings(compression: str) -> tuple[Optional[str], Optional[int]]:
    """
    Get h5py compression settings from compression string.
    
    Parameters
    ----------
    compression : str
        Compression type: 'gzip-4', 'gzip-1', 'lzf', or 'none'.
        
    Returns
    -------
    tuple[str | None, int | None]
        (compression_type, compression_opts) for h5py.create_dataset
    """
    compression = compression.lower().strip()
    
    if compression == 'none':
        return None, None
    elif compression == 'lzf':
        return 'lzf', None
    elif compression.startswith('gzip'):
        # Parse gzip-N format
        if '-' in compression:
            try:
                level = int(compression.split('-')[1])
                level = max(1, min(9, level))  # Clamp to valid range
            except (ValueError, IndexError):
                level = 1
        else:
            level = 1  # Default gzip level
        return 'gzip', level
    else:
        # Unknown compression, default to gzip-1
        return 'gzip', 1


def _save_h5(
    data: np.ndarray,
    output_path: Path,
    max_val: float,
    element_size_um: tuple[float, float, float],
    metadata: dict,
    channel_name: str,
    compression: str = 'gzip-1',
) -> None:
    """Save data as HDF5 file with metadata attributes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comp_type, comp_opts = _get_compression_settings(compression)
    
    with h5py.File(output_path, 'w') as f:
        chunk_size = tuple(min(64, s) for s in data.shape)
        
        # Build create_dataset kwargs based on compression
        ds_kwargs = {
            'data': data,
            'chunks': chunk_size,
        }
        if comp_type is not None:
            ds_kwargs['compression'] = comp_type
            if comp_opts is not None:
                ds_kwargs['compression_opts'] = comp_opts
        
        ds = f.create_dataset('data', **ds_kwargs)
        
        ds.attrs['maxVal'] = np.array([max_val], dtype=np.float32)
        ds.attrs['element_size_um'] = np.array(element_size_um, dtype=np.float32)
        ds.attrs['channelNames'] = np.array([channel_name], dtype=object)
        
        for key, value in metadata.items():
            if key == 'channelNames':
                continue
            if isinstance(value, (int, float)):
                ds.attrs[key] = np.array([float(value)], dtype=np.float32)
            elif isinstance(value, str):
                ds.attrs[key] = np.array([value], dtype=object)
            elif isinstance(value, (list, tuple)):
                ds.attrs[key] = np.array([float(v) for v in value], dtype=np.float32)


def _save_tiff(
    data: np.ndarray,
    output_path: Path,
    pixel_size_um: float,
    z_step_um: float,
) -> None:
    """Save data as TIFF file with ImageJ-compatible metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    imagej_metadata = {
        'unit': 'micron',
        'spacing': z_step_um,
        'slices': data.shape[0],
    }
    
    resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
    
    with TiffWriter(output_path, imagej=True) as tif:
        tif.write(
            data,
            resolution=resolution,
            resolutionunit='MICROMETER',
            metadata=imagej_metadata,
        )


@dataclass
class ChannelTask:
    """Task for processing a single channel."""
    channel_name: str
    channel_data: np.ndarray
    formats: dict  # {'h5_16bit': Path, 'h5_8bit': Path, ...}
    element_size_um: tuple
    metadata: dict
    pixel_size_um: float
    z_step: float
    overwrite: bool
    compression: str = 'gzip-1'
    rescale: bool = True


def _process_channel_task(task: ChannelTask) -> dict:
    """
    Process a single channel: normalize and save to requested formats.
    Thread-safe function for parallel processing.
    """
    results = {'processed': [], 'skipped': []}
    channel_name = task.channel_name
    channel_data = task.channel_data.astype(np.float32)
    
    # H5 16-bit
    if 'h5_16bit' in task.formats:
        out_path = task.formats['h5_16bit']
        if out_path.exists() and not task.overwrite:
            results['skipped'].append(str(out_path))
        else:
            if task.rescale:
                data_16bit, max_val = _normalize_to_16bit(channel_data)
            else:
                max_val = float(np.max(channel_data))
                data_16bit = np.clip(channel_data, 0, 65535).astype(np.uint16)
            _save_h5(data_16bit, out_path, max_val, task.element_size_um, task.metadata, channel_name, task.compression)
            results['processed'].append(str(out_path))
    
    # H5 8-bit
    if 'h5_8bit' in task.formats:
        out_path = task.formats['h5_8bit']
        if out_path.exists() and not task.overwrite:
            results['skipped'].append(str(out_path))
        else:
            data_8bit, max_val = _normalize_to_8bit(channel_data)
            _save_h5(data_8bit, out_path, max_val, task.element_size_um, task.metadata, channel_name, task.compression)
            results['processed'].append(str(out_path))
    
    # TIFF 16-bit
    if 'tiff_16bit' in task.formats:
        out_path = task.formats['tiff_16bit']
        if out_path.exists() and not task.overwrite:
            results['skipped'].append(str(out_path))
        else:
            if task.rescale:
                data_16bit, _ = _normalize_to_16bit(channel_data)
            else:
                data_16bit = np.clip(channel_data, 0, 65535).astype(np.uint16)
            _save_tiff(data_16bit, out_path, task.pixel_size_um, task.z_step)
            results['processed'].append(str(out_path))
    
    # TIFF 8-bit
    if 'tiff_8bit' in task.formats:
        out_path = task.formats['tiff_8bit']
        if out_path.exists() and not task.overwrite:
            results['skipped'].append(str(out_path))
        else:
            data_8bit, _ = _normalize_to_8bit(channel_data)
            _save_tiff(data_8bit, out_path, task.pixel_size_um, task.z_step)
            results['processed'].append(str(out_path))
    
    return results


def process_nd2_to_h5(
    nd2_path: str | Path,
    output_dir: str | Path,
    z_step: float = 0.4,
    h5_16bit: Optional[list[str]] = None,
    h5_8bit: Optional[list[str]] = None,
    tiff_16bit: Optional[list[str]] = None,
    tiff_8bit: Optional[list[str]] = None,
    overwrite: bool = False,
    channel_workers: int = 1,
    compression: str = 'gzip-1',
    rescale: bool = True,
    progress_callback: Optional[callable] = None,
) -> dict[str, list[str]]:
    """
    Process a single ND2 file to H5/TIFF formats.
    
    Channels with duplicate names are automatically averaged.
    
    Parameters
    ----------
    nd2_path : str or Path
        Path to the ND2 file.
    output_dir : str or Path
        Output directory for processed files.
    z_step : float, optional
        Z-step size in microns. Default is 0.4.
    h5_16bit : list of str, optional
        Channels to save as 16-bit HDF5. Use ['*'] for all channels.
    h5_8bit : list of str, optional
        Channels to save as 8-bit HDF5. Use ['*'] for all channels.
    tiff_16bit : list of str, optional
        Channels to save as 16-bit TIFF. Use ['*'] for all channels.
    tiff_8bit : list of str, optional
        Channels to save as 8-bit TIFF. Use ['*'] for all channels.
    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    channel_workers : int, optional
        Number of parallel workers for channel processing. Default is 1.
    compression : str, optional
        H5 compression type: 'gzip-4', 'gzip-1', 'lzf', or 'none'. Default is 'gzip-1'.
    progress_callback : callable, optional
        Callback function(channel_name, status) for progress updates.
        
    Returns
    -------
    dict[str, list[str]]
        Dictionary with 'processed' and 'skipped' lists of output file paths.
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    
    h5_16bit = h5_16bit or []
    h5_8bit = h5_8bit or []
    tiff_16bit = tiff_16bit or []
    tiff_8bit = tiff_8bit or []
    
    if not any([h5_16bit, h5_8bit, tiff_16bit, tiff_8bit]):
        return {'processed': [], 'skipped': []}
    
    basename = _parse_basename(nd2_path)
    
    # Read ND2 file
    with nd2.ND2File(nd2_path) as f:
        data = f.asarray()
        
        channel_names = []
        if f.metadata and f.metadata.channels:
            channel_names = [ch.channel.name for ch in f.metadata.channels]
        
        pixel_size_um = 1.0
        if f.metadata and f.metadata.channels:
            cal = f.metadata.channels[0].volume.axesCalibration
            if cal:
                pixel_size_um = float(cal[0])
        
        sizes = f.sizes
        metadata = {
            'sizeX': sizes.get('X', data.shape[-1]),
            'sizeY': sizes.get('Y', data.shape[-2]),
            'sizeZ': sizes.get('Z', data.shape[0] if data.ndim >= 3 else 1),
            'sizeC': sizes.get('C', len(channel_names)),
            'sizeT': sizes.get('T', 1),
            'bitDepth': 16,
            'pixelSize': pixel_size_um,
        }
    
    # Group channels by name
    channel_groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(channel_names):
        channel_groups[name].append(idx)
    
    unique_channels = list(channel_groups.keys())
    element_size_um = (float(z_step), pixel_size_um, pixel_size_um)
    
    # Expand wildcards
    def expand_channels(channel_list: list[str]) -> list[str]:
        if '*' in channel_list:
            return unique_channels.copy()
        return [ch for ch in channel_list if ch in unique_channels]
    
    h5_16bit_channels = expand_channels(h5_16bit)
    h5_8bit_channels = expand_channels(h5_8bit)
    tiff_16bit_channels = expand_channels(tiff_16bit)
    tiff_8bit_channels = expand_channels(tiff_8bit)
    
    all_requested_channels = set(
        h5_16bit_channels + h5_8bit_channels + 
        tiff_16bit_channels + tiff_8bit_channels
    )
    
    if not all_requested_channels:
        return {'processed': [], 'skipped': []}
    
    # Create output directories
    dirs = {}
    if h5_16bit_channels:
        dirs['h5_16bit'] = output_dir / 'h5_16bit'
        dirs['h5_16bit'].mkdir(parents=True, exist_ok=True)
    if h5_8bit_channels:
        dirs['h5_8bit'] = output_dir / 'h5_8bit'
        dirs['h5_8bit'].mkdir(parents=True, exist_ok=True)
    if tiff_16bit_channels:
        dirs['tiff_16bit'] = output_dir / 'tiff_16bit'
        dirs['tiff_16bit'].mkdir(parents=True, exist_ok=True)
    if tiff_8bit_channels:
        dirs['tiff_8bit'] = output_dir / 'tiff_8bit'
        dirs['tiff_8bit'].mkdir(parents=True, exist_ok=True)
    
    # Pre-extract and average all channels
    channel_data_map = {}
    for channel_name in all_requested_channels:
        indices = channel_groups[channel_name]
        
        if len(indices) == 1:
            channel_data = data[:, indices[0], :, :]
        else:
            channel_data = data[:, indices, :, :].mean(axis=1)
        
        # Transpose to (Z, X, Y) to match MATLAB output
        channel_data_map[channel_name] = np.transpose(channel_data, (0, 2, 1))
    
    # Build channel tasks
    tasks = []
    for channel_name in all_requested_channels:
        out_name = f"{basename}_channel-{channel_name}"
        formats = {}
        
        if channel_name in h5_16bit_channels:
            formats['h5_16bit'] = dirs['h5_16bit'] / f"{out_name}.h5"
        if channel_name in h5_8bit_channels:
            formats['h5_8bit'] = dirs['h5_8bit'] / f"{out_name}.h5"
        if channel_name in tiff_16bit_channels:
            formats['tiff_16bit'] = dirs['tiff_16bit'] / f"{out_name}.tif"
        if channel_name in tiff_8bit_channels:
            formats['tiff_8bit'] = dirs['tiff_8bit'] / f"{out_name}.tif"
        
        tasks.append(ChannelTask(
            channel_name=channel_name,
            channel_data=channel_data_map[channel_name],
            formats=formats,
            element_size_um=element_size_um,
            metadata=metadata,
            pixel_size_um=pixel_size_um,
            z_step=z_step,
            overwrite=overwrite,
            compression=compression,
            rescale=rescale,
        ))
    
    results = {'processed': [], 'skipped': []}
    
    if channel_workers <= 1:
        # Sequential processing
        for task in tasks:
            if progress_callback:
                progress_callback(task.channel_name, 'processing')
            result = _process_channel_task(task)
            results['processed'].extend(result['processed'])
            results['skipped'].extend(result['skipped'])
            if progress_callback:
                progress_callback(task.channel_name, 'done')
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=channel_workers) as executor:
            futures = {executor.submit(_process_channel_task, task): task for task in tasks}
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results['processed'].extend(result['processed'])
                    results['skipped'].extend(result['skipped'])
                    if progress_callback:
                        progress_callback(task.channel_name, 'done')
                except Exception as e:
                    _safe_print(f"Error processing channel {task.channel_name}: {e}")
    
    return results


def _process_single_file_wrapper(args: tuple) -> dict:
    """Worker function for parallel file processing."""
    (nd2_path, output_dir, z_step, h5_16bit, h5_8bit, tiff_16bit, tiff_8bit,
     overwrite, channel_workers, compression, rescale) = args
    try:
        return process_nd2_to_h5(
            nd2_path, output_dir, z_step,
            h5_16bit, h5_8bit, tiff_16bit, tiff_8bit,
            overwrite, channel_workers, compression, rescale
        )
    except Exception as e:
        _safe_print(f"Error processing {nd2_path}: {e}")
        return {'processed': [], 'skipped': [], 'error': str(e)}


def process_nd2_folder(
    nd2_folder: str | Path,
    output_dir: str | Path,
    ref_step: float = 0.2,
    reg_step: float = 0.4,
    h5_16bit: Optional[list[str]] = None,
    h5_8bit: Optional[list[str]] = None,
    tiff_16bit: Optional[list[str]] = None,
    tiff_8bit: Optional[list[str]] = None,
    workers: Optional[int] = None,
    channel_workers: Optional[int] = None,
    compression: str = 'gzip-1',
    max_memory: Optional[str] = None,
    overwrite: bool = False,
    rescale: bool = True,
) -> dict[str, int]:
    """
    Batch process all ND2 files in a folder structure.
    
    Parameters
    ----------
    nd2_folder : str or Path
        Root folder containing round subdirectories with ND2 files.
    output_dir : str or Path
        Output directory for processed files.
    ref_step : float, optional
        Z-step for reference round (round00). Default is 0.2.
    reg_step : float, optional
        Z-step for registration rounds. Default is 0.4.
    h5_16bit : list of str, optional
        Channels to save as 16-bit HDF5.
    h5_8bit : list of str, optional
        Channels to save as 8-bit HDF5.
    tiff_16bit : list of str, optional
        Channels to save as 16-bit TIFF.
    tiff_8bit : list of str, optional
        Channels to save as 8-bit TIFF.
    workers : int, optional
        Number of parallel file workers. Default: auto-detect.
    channel_workers : int, optional
        Number of parallel channel workers per file. Default: auto-detect.
    compression : str, optional
        H5 compression type: 'gzip-4', 'gzip-1', 'lzf', or 'none'. Default is 'gzip-1'.
    max_memory : str, optional
        Maximum memory to use (e.g., '64G', '256GB'). Default: 80% of available.
    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
        
    Returns
    -------
    dict[str, int]
        Summary with counts of processed, skipped, and error files.
    """
    nd2_folder = Path(nd2_folder)
    output_dir = Path(output_dir)
    
    # Find all ND2 files
    nd2_files = []
    
    direct_nd2 = list(nd2_folder.glob('*.nd2'))
    if direct_nd2:
        for nd2_path in sorted(direct_nd2):
            is_round00 = 'round00' in nd2_folder.name
            z_step = ref_step if is_round00 else reg_step
            nd2_files.append((nd2_path, z_step))
    else:
        for round_dir in sorted(nd2_folder.iterdir()):
            if not round_dir.is_dir():
                continue
            
            round_nd2_files = list(round_dir.glob('*.nd2'))
            if not round_nd2_files:
                continue
            
            is_round00 = 'round00' in round_dir.name
            z_step = ref_step if is_round00 else reg_step
            
            for nd2_path in sorted(round_nd2_files):
                nd2_files.append((nd2_path, z_step))
    
    if not nd2_files:
        print(f"No ND2 files found in {nd2_folder}")
        return {'processed': 0, 'skipped': 0, 'errors': 0}
    
    # Count unique channels requested
    all_channels = set()
    for ch_list in [h5_16bit, h5_8bit, tiff_16bit, tiff_8bit]:
        if ch_list:
            if '*' in ch_list:
                all_channels.add('*')
            else:
                all_channels.update(ch_list)
    n_channels = 5 if '*' in all_channels else len(all_channels)
    
    # Auto-detect settings if not specified
    max_memory_gb = None
    if max_memory:
        max_memory_gb = _parse_memory_string(max_memory)
    
    auto_workers, auto_channel_workers = _auto_detect_settings(max_memory_gb, n_channels)
    
    if workers is None:
        workers = auto_workers
    if channel_workers is None:
        channel_workers = auto_channel_workers
    
    # Memory warning
    estimated_memory_gb = workers * _estimate_memory_per_file_gb(n_channels)
    available_memory_gb = _get_available_memory_gb()
    
    print(f"\n{'='*60}")
    print(f"ND2 to H5/TIFF Processor")
    print(f"{'='*60}")
    print(f"Input folder:  {nd2_folder}")
    print(f"Output folder: {output_dir}")
    print(f"Files to process: {len(nd2_files)}")
    print(f"\nParallelism settings:")
    print(f"  File workers:    {workers}")
    print(f"  Channel workers: {channel_workers}")
    print(f"  Total threads:   {workers * channel_workers}")
    print(f"\nMemory estimate:")
    print(f"  Available: {available_memory_gb:.1f} GB")
    print(f"  Estimated: {estimated_memory_gb:.1f} GB")
    
    if estimated_memory_gb > available_memory_gb * 0.9:
        print(f"\n  WARNING: Estimated memory usage ({estimated_memory_gb:.1f} GB) exceeds")
        print(f"           90% of available memory ({available_memory_gb:.1f} GB).")
        print(f"           Consider reducing --workers or --channel-workers.")
    
    print(f"\nCompression: {compression}")
    print(f"Rescale 16-bit: {rescale}")
    print(f"{'='*60}\n")
    
    # Prepare task arguments
    task_args = [
        (nd2_path, output_dir, z_step, h5_16bit, h5_8bit, tiff_16bit, tiff_8bit,
         overwrite, channel_workers, compression, rescale)
        for nd2_path, z_step in nd2_files
    ]
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    # Create progress bars if tqdm is available
    if HAS_TQDM:
        file_pbar = tqdm(total=len(nd2_files), desc="Files", unit="file", position=0)
        channel_pbar = tqdm(total=len(nd2_files) * n_channels, desc="Channels", unit="ch", position=1)
    
    def update_progress(result):
        nonlocal total_processed, total_skipped, total_errors
        total_processed += len(result.get('processed', []))
        total_skipped += len(result.get('skipped', []))
        if 'error' in result:
            total_errors += 1
        if HAS_TQDM:
            file_pbar.update(1)
            channel_pbar.update(len(result.get('processed', [])) + len(result.get('skipped', [])))
    
    if workers <= 1:
        # Sequential processing
        for args in task_args:
            result = _process_single_file_wrapper(args)
            update_progress(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_file_wrapper, args): args[0] for args in task_args}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    update_progress(result)
                except Exception as e:
                    nd2_path = futures[future]
                    print(f"Error processing {nd2_path}: {e}")
                    total_errors += 1
                    if HAS_TQDM:
                        file_pbar.update(1)
    
    if HAS_TQDM:
        file_pbar.close()
        channel_pbar.close()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Processed: {total_processed} files")
    print(f"  Skipped:   {total_skipped} files")
    print(f"  Errors:    {total_errors} files")
    print(f"{'='*60}")
    
    return {
        'processed': total_processed,
        'skipped': total_skipped,
        'errors': total_errors,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Process ND2 files to H5/TIFF formats with parallel processing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process folder with specific channel formats
  python -m ifish_tools.nd2_processor /path/to/nd2/folder /path/to/output \\
      --h5-16bit "Cy3 Nar" "Cy5" "mCherry Nar" \\
      --h5-8bit "DAPI" "FITC-GFP"

  # Process all channels as h5_16bit with parallel processing
  python -m ifish_tools.nd2_processor /path/to/nd2/folder /path/to/output \\
      --h5-16bit "*" --workers 4 --channel-workers 5

  # Limit memory usage
  python -m ifish_tools.nd2_processor /path/to/nd2/folder /path/to/output \\
      --h5-16bit "*" --max-memory 64G

  # Process single ND2 file
  python -m ifish_tools.nd2_processor /path/to/file.nd2 /path/to/output \\
      --z-step 0.4 --h5-16bit "*"

  # Use faster compression (lzf) for speed, or gzip-4 for smaller files
  python -m ifish_tools.nd2_processor /path/to/nd2/folder /path/to/output \\
      --h5-16bit "*" --compression lzf
        """
    )
    
    parser.add_argument('input', type=str,
                        help='Input ND2 file or folder containing ND2 files')
    parser.add_argument('output', type=str,
                        help='Output directory for processed files')
    
    parser.add_argument('--ref-step', type=float, default=0.2,
                        help='Z-step for reference round (round00). Default: 0.2')
    parser.add_argument('--reg-step', type=float, default=0.4,
                        help='Z-step for registration rounds. Default: 0.4')
    parser.add_argument('--z-step', type=float, default=None,
                        help='Z-step for single file processing (overrides ref/reg-step)')
    
    parser.add_argument('--h5-16bit', nargs='*', default=None,
                        help='Channels to save as 16-bit HDF5. Use "*" for all.')
    parser.add_argument('--h5-8bit', nargs='*', default=None,
                        help='Channels to save as 8-bit HDF5. Use "*" for all.')
    parser.add_argument('--tiff-16bit', nargs='*', default=None,
                        help='Channels to save as 16-bit TIFF. Use "*" for all.')
    parser.add_argument('--tiff-8bit', nargs='*', default=None,
                        help='Channels to save as 8-bit TIFF. Use "*" for all.')
    
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel file workers. Default: auto-detect')
    parser.add_argument('--channel-workers', type=int, default=None,
                        help='Number of parallel channel workers per file. Default: auto-detect')
    parser.add_argument('--max-memory', type=str, default=None,
                        help='Maximum memory to use (e.g., "64G", "256GB"). Default: 80%% available')
    parser.add_argument('--compression', type=str, default='gzip-1',
                        choices=['gzip-4', 'gzip-1', 'lzf', 'none'],
                        help='H5 compression type. Default: gzip-1. Options: gzip-4 (slow, small), gzip-1 (balanced), lzf (fast, larger), none (fastest, largest)')
    
    parser.add_argument('--rescale', action=argparse.BooleanOptionalAction, default=True,
                        help='Rescale 16-bit data to full range (default: True). Use --no-rescale to save raw values.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.nd2':
        # Single file processing
        z_step = args.z_step if args.z_step is not None else args.reg_step
        channel_workers = args.channel_workers or 1
        
        print(f"Processing single file: {input_path.name}")
        result = process_nd2_to_h5(
            input_path,
            args.output,
            z_step=z_step,
            h5_16bit=args.h5_16bit,
            h5_8bit=args.h5_8bit,
            tiff_16bit=args.tiff_16bit,
            tiff_8bit=args.tiff_8bit,
            overwrite=args.overwrite,
            channel_workers=channel_workers,
            compression=args.compression,
            rescale=args.rescale,
        )
        print(f"Processed: {len(result['processed'])}, Skipped: {len(result['skipped'])}")
        
    elif input_path.is_dir():
        # Folder processing
        process_nd2_folder(
            input_path,
            args.output,
            ref_step=args.ref_step,
            reg_step=args.reg_step,
            h5_16bit=args.h5_16bit,
            h5_8bit=args.h5_8bit,
            tiff_16bit=args.tiff_16bit,
            tiff_8bit=args.tiff_8bit,
            workers=args.workers,
            channel_workers=args.channel_workers,
            compression=args.compression,
            max_memory=args.max_memory,
            overwrite=args.overwrite,
            rescale=args.rescale,
        )
    else:
        print(f"Error: {input_path} is not a valid ND2 file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
