"""
Puncta registration pipeline for iFISH data.

This module implements the complete registration pipeline that matches the MATLAB
dir_regis_puncta.m workflow:
1. Load raw RS-FISH puncta (0-based coordinates)
2. Apply affine transformation
3. Clip to bounding box
4. Apply non-rigid demons displacement field
5. Output registered puncta

CRITICAL AXIS CONVENTIONS:
- H5 files store data in C-order: (Z, sizeX, sizeY) due to nd2_processor transpose
- MATLAB reads H5 in reversed order: (sizeY, sizeX, Z)
- RS-FISH CSV: x, y, z columns (0-based pixel coordinates)
- bbox.x → MATLAB dim1 (sizeY), bbox.y → MATLAB dim2 (sizeX)
- Displacement field D: H5 C-order (3, Nz, Nx, Ny) → MATLAB sees (Ny, Nx, Nz, 3)
- D components: 0=X-disp (cols), 1=Y-disp (rows), 2=Z-disp
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.io
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

# Default spot radius for registered puncta (µm, typical RS-FISH spot radius)
DEFAULT_SPOT_RADIUS = 0.015


def load_affine_transform(corr_h5_path: Path) -> np.ndarray:
    """
    Load affine transformation matrix from correlation H5 file.
    
    Args:
        corr_h5_path: Path to H5 file containing transformation matrix
        
    Returns:
        T_forward: 4x4 transformation matrix (transposed to match MATLAB convention)
        
    Notes:
        MATLAB uses affinetform3d(T') where T' is the transpose.
        We read T and return T.T to match this convention.
    """
    try:
        with h5py.File(corr_h5_path, 'r') as f:
            T = f['/data'].attrs['T'][:]  # Shape (4, 4)
    except Exception as e:
        logger.error(f"Failed to load affine transform from {corr_h5_path}: {e}")
        raise
    
    logger.debug(f"Loaded affine transform from {corr_h5_path}")
    return T.T  # Transpose to match MATLAB convention


def transform_points_forward(T_forward: np.ndarray, x: np.ndarray, 
                             y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply affine transformation to points using row-vector convention.
    
    Args:
        T_forward: 4x4 transformation matrix
        x, y, z: Coordinate arrays (same length)
        
    Returns:
        x_out, y_out, z_out: Transformed coordinates
        
    Notes:
        Uses row-vector × matrix convention: [x, y, z, 1] @ T_forward
        This matches MATLAB's transformPointsForward behavior.
    """
    # Build homogeneous coordinates (N, 4)
    N = len(x)
    pts_in = np.column_stack([x, y, z, np.ones(N)])
    
    # Apply transform: row-vector × matrix
    pts_out = pts_in @ T_forward
    
    return pts_out[:, 0], pts_out[:, 1], pts_out[:, 2]


def load_displacement_field(D_h5_path: Path) -> np.ndarray:
    """
    Load displacement field from H5 file and convert to MATLAB layout.
    
    Args:
        D_h5_path: Path to displacement field H5 file
        
    Returns:
        D: Displacement field in MATLAB layout (Ny, Nx, Nz, 3)
        
    Notes:
        H5 stores as C-order (3, Nz, Nx, Ny)
        MATLAB expects (Ny, Nx, Nz, 3)
        Transpose: (3, Nz, Nx, Ny) → (Ny, Nx, Nz, 3)
    """
    try:
        with h5py.File(D_h5_path, 'r') as f:
            D_c = f['/data'][:]  # Shape: (3, Nz, Nx, Ny)
    except Exception as e:
        logger.error(f"Failed to load displacement field from {D_h5_path}: {e}")
        raise
    
    # Transpose to MATLAB layout: (3, Nz, Nx, Ny) → (Ny, Nx, Nz, 3)
    D_matlab = D_c.transpose(3, 2, 1, 0)
    
    logger.debug(f"Loaded displacement field from {D_h5_path}, shape: {D_matlab.shape}")
    return D_matlab


def resize_displacement_field(D: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize displacement field to target size using cubic interpolation.
    
    Args:
        D: Displacement field (Ny, Nx, Nz, 3)
        target_size: Target size (Ny_new, Nx_new, Nz_new)
        
    Returns:
        D_resized: Resized displacement field with scaled displacements
        
    Notes:
        Uses scipy.ndimage.zoom with order=3 (cubic interpolation).
        Displacement values are scaled by the zoom factor for each dimension.
    """
    Ny_old, Nx_old, Nz_old, _ = D.shape
    Ny_new, Nx_new, Nz_new = target_size
    
    # Calculate zoom factors
    zoom_y = Ny_new / Ny_old
    zoom_x = Nx_new / Nx_old
    zoom_z = Nz_new / Nz_old
    
    # Resize each component separately
    D_resized = np.zeros((Ny_new, Nx_new, Nz_new, 3))
    
    for i in range(3):
        D_resized[:, :, :, i] = zoom(D[:, :, :, i], 
                                     (zoom_y, zoom_x, zoom_z), 
                                     order=3)
    
    # Scale displacement values
    D_resized[:, :, :, 0] *= zoom_x  # X displacement
    D_resized[:, :, :, 1] *= zoom_y  # Y displacement
    D_resized[:, :, :, 2] *= zoom_z  # Z displacement
    
    logger.debug(f"Resized displacement field from {D.shape} to {D_resized.shape}")
    return D_resized


def apply_demons_displacement(puncta: np.ndarray, Dfull: np.ndarray) -> np.ndarray:
    """
    Apply demons displacement field to puncta coordinates.
    
    Args:
        puncta: Array of shape (N, >=3) with columns [x, y, z, ...]
                x, y, z are 1-based coordinates in cropped space
        Dfull: Displacement field (Ny, Nx, Nz, 3) in MATLAB layout
        
    Returns:
        puncta_registered: Same shape as input with updated x, y, z
        
    Notes:
        MATLAB queries interpolator as (y, x, z) with 1-based indexing.
        Displacement is SUBTRACTED: x' = x - dx, y' = y - dy, z' = z - dz
    """
    Ny, Nx, Nz, _ = Dfull.shape
    
    # Create 1-based grids for interpolation (MATLAB convention)
    y_grid = np.arange(1, Ny + 1, dtype=float)
    x_grid = np.arange(1, Nx + 1, dtype=float)
    z_grid = np.arange(1, Nz + 1, dtype=float)
    
    # Create interpolators for each displacement component
    # Query points will be (y, x, z) in MATLAB convention
    interp_dx = RegularGridInterpolator((y_grid, x_grid, z_grid), 
                                        Dfull[:, :, :, 0], 
                                        bounds_error=False, 
                                        fill_value=0.0)
    interp_dy = RegularGridInterpolator((y_grid, x_grid, z_grid), 
                                        Dfull[:, :, :, 1], 
                                        bounds_error=False, 
                                        fill_value=0.0)
    interp_dz = RegularGridInterpolator((y_grid, x_grid, z_grid), 
                                        Dfull[:, :, :, 2], 
                                        bounds_error=False, 
                                        fill_value=0.0)
    
    # Extract coordinates (1-based)
    x = puncta[:, 0]
    y = puncta[:, 1]
    z = puncta[:, 2]
    
    # Query interpolators with (y, x, z) order
    query_pts = np.column_stack([y, x, z])
    dx = interp_dx(query_pts)
    dy = interp_dy(query_pts)
    dz = interp_dz(query_pts)
    
    # Subtract displacements (MATLAB convention)
    puncta_registered = puncta.copy()
    puncta_registered[:, 0] = x - dx
    puncta_registered[:, 1] = y - dy
    puncta_registered[:, 2] = z - dz
    
    logger.debug(f"Applied demons displacement to {len(puncta)} puncta")
    return puncta_registered


def load_bbox(bbox_mat_path: Path, var_name: str = 'bbox') -> Dict[str, int]:
    """
    Load bounding box from MATLAB .mat file.
    
    Args:
        bbox_mat_path: Path to .mat file
        var_name: Variable name in .mat file (default: 'bbox')
        
    Returns:
        Dictionary with keys: xmin, xmax, ymin, ymax, zmin, zmax
        
    Notes:
        MATLAB bbox structure uses 1-based indexing.
        We preserve the values as-is for compatibility.
    """
    try:
        mat = scipy.io.loadmat(bbox_mat_path)
        bbox_struct = mat[var_name]
        
        # Extract fields from MATLAB struct
        bbox = {
            'xmin': int(bbox_struct['xmin'][0, 0][0, 0]),
            'xmax': int(bbox_struct['xmax'][0, 0][0, 0]),
            'ymin': int(bbox_struct['ymin'][0, 0][0, 0]),
            'ymax': int(bbox_struct['ymax'][0, 0][0, 0]),
            'zmin': int(bbox_struct['zmin'][0, 0][0, 0]),
            'zmax': int(bbox_struct['zmax'][0, 0][0, 0]),
        }
    except Exception as e:
        logger.error(f"Failed to load bbox from {bbox_mat_path}: {e}")
        raise
    
    logger.debug(f"Loaded bbox from {bbox_mat_path}: {bbox}")
    return bbox


def get_cropped_image_size(ref_h5_path: Path, bbox: Dict[str, int]) -> Tuple[int, int, int]:
    """
    Calculate cropped image size from bounding box.
    
    Args:
        ref_h5_path: Path to reference H5 file (not used, kept for compatibility)
        bbox: Bounding box dictionary with xmin, xmax, ymin, ymax, zmin, zmax
        
    Returns:
        (Ny, Nx, Nz): Size of cropped region
        
    Notes:
        Due to MATLAB axis conventions:
        - Ny = bbox.xmax - bbox.xmin + 1 (bbox.x → MATLAB dim1)
        - Nx = bbox.ymax - bbox.ymin + 1 (bbox.y → MATLAB dim2)
        - Nz = bbox.zmax - bbox.zmin + 1
    """
    Ny = bbox['xmax'] - bbox['xmin'] + 1
    Nx = bbox['ymax'] - bbox['ymin'] + 1
    Nz = bbox['zmax'] - bbox['zmin'] + 1
    
    logger.debug(f"Calculated cropped size: Ny={Ny}, Nx={Nx}, Nz={Nz}")
    return (Ny, Nx, Nz)


def register_puncta_for_round(
    puncta_csv: Path,
    T_forward: np.ndarray,
    Dfull: np.ndarray,
    bbox_reg: Dict[str, int],
    pixel_size: float,
    ref_step: float,
    output_path: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Register puncta for one round using full pipeline.
    
    Pipeline steps (matching MATLAB dir_regis_puncta.m lines 150-202):
    1. Read raw puncta (x, y, z are 0-based)
    2. Add 1 to convert to 1-based
    3. Convert to world coordinates
    4. Apply affine transformation
    5. Convert back to pixel coordinates
    6. Clip to bbox_reg
    7. Shift to cropped coordinates
    8. Apply demons displacement
    9. Remove out-of-bounds puncta
    
    Args:
        puncta_csv: Path to RS-FISH CSV file
        T_forward: 4x4 affine transformation matrix
        Dfull: Displacement field (Ny, Nx, Nz, 3) resized to cropped size
        bbox_reg: Registration bounding box
        pixel_size: Pixel size in microns (e.g., 0.108)
        ref_step: Z-step size in microns (e.g., 0.4)
        output_path: Optional path to save registered CSV
        
    Returns:
        DataFrame with registered puncta, or None if no puncta remain
    """
    logger.info(f"Processing {puncta_csv}")
    
    # Step 1: Read raw puncta (0-based)
    try:
        df = pd.read_csv(puncta_csv)
    except Exception as e:
        logger.error(f"Failed to read {puncta_csv}: {e}")
        return None
    
    if len(df) == 0:
        logger.warning(f"Empty CSV: {puncta_csv}")
        return None
    
    # Ensure required columns exist
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        logger.error(f"Missing required columns in {puncta_csv}")
        return None
    
    # Step 2: Convert to 1-based (MATLAB convention)
    x_intr = df['x'].values + 1
    y_intr = df['y'].values + 1
    z_intr = df['z'].values + 1
    
    # Step 3: Convert to world coordinates
    x_world = x_intr * pixel_size
    y_world = y_intr * pixel_size
    z_world = z_intr * ref_step
    
    # Step 4: Apply affine transformation
    x_ref, y_ref, z_ref = transform_points_forward(T_forward, x_world, y_world, z_world)
    
    # Step 5: Convert back to pixel coordinates
    x_ref = x_ref / pixel_size
    y_ref = y_ref / pixel_size
    z_ref = z_ref / ref_step
    
    # Step 6: Clip to bbox_reg (CRITICAL: x vs bbox.y, y vs bbox.x swap!)
    mask = (
        (x_ref >= bbox_reg['ymin']) & (x_ref <= bbox_reg['ymax']) &
        (y_ref >= bbox_reg['xmin']) & (y_ref <= bbox_reg['xmax']) &
        (z_ref >= bbox_reg['zmin']) & (z_ref <= bbox_reg['zmax'])
    )
    
    x_ref = x_ref[mask]
    y_ref = y_ref[mask]
    z_ref = z_ref[mask]
    
    if len(x_ref) == 0:
        logger.warning(f"No puncta within bbox after affine: {puncta_csv}")
        return None
    
    # Step 7: Shift to cropped coordinates (1-based)
    x_crop = x_ref - bbox_reg['ymin'] + 1
    y_crop = y_ref - bbox_reg['xmin'] + 1
    z_crop = z_ref - bbox_reg['zmin'] + 1
    
    # Step 8: Apply demons displacement
    # Build puncta array with extra columns for intensity
    intensity = df['intensity'].values[mask] if 'intensity' in df.columns else np.ones(len(x_crop))
    puncta_array = np.column_stack([x_crop, y_crop, z_crop, intensity])
    puncta_registered = apply_demons_displacement(puncta_array, Dfull)
    
    # Step 9: Remove out-of-bounds puncta
    Ny, Nx, Nz, _ = Dfull.shape
    final_mask = (
        (puncta_registered[:, 0] >= 1) & (puncta_registered[:, 0] <= Nx) &
        (puncta_registered[:, 1] >= 1) & (puncta_registered[:, 1] <= Ny) &
        (puncta_registered[:, 2] >= 1) & (puncta_registered[:, 2] <= Nz)
    )
    
    puncta_final = puncta_registered[final_mask]
    
    if len(puncta_final) == 0:
        logger.warning(f"No puncta remaining after demons: {puncta_csv}")
        return None
    
    # Create output DataFrame with MATLAB-compatible format
    # Convert back to 0-based for output
    result_df = pd.DataFrame({
        'x': puncta_final[:, 0] - 1,
        'y': puncta_final[:, 1] - 1,
        'z': puncta_final[:, 2] - 1,
        't': 0,
        'r': DEFAULT_SPOT_RADIUS,
        'cell_id': '',
        'parent_id': '',
        'intensity': puncta_final[:, 3]
    })
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(result_df)} registered puncta to {output_path}")
    
    return result_df


def register_brain(
    brain_id: str,
    puncta_dir: Path,
    res_dir: Path,
    h5_dir: Path,
    output_dir: Path,
    ref_step: float = 0.4,
    cpd: str = "Affine"
):
    """
    Register all puncta for one brain across all rounds.
    
    Args:
        brain_id: Brain identifier (e.g., "brain0")
        puncta_dir: Directory containing RS-FISH puncta CSVs
        res_dir: Directory containing registration results
        h5_dir: Directory containing H5 image files
        output_dir: Output directory for registered puncta
        ref_step: Z-step size in microns (default: 0.4)
        cpd: Registration type (default: "Affine")
    """
    logger.info(f"Starting registration for {brain_id}")
    
    brain_res_dir = res_dir / brain_id
    regdemon_dir = brain_res_dir / "regdemon"
    corr_dir = brain_res_dir / "corr"
    ref_dir = brain_res_dir / "ref"
    
    # Load bounding boxes
    bbox_ref_path = ref_dir / "bbox_ref.mat"
    bbox_reg_path = ref_dir / "bbox_reg.mat"
    
    if not bbox_ref_path.exists() or not bbox_reg_path.exists():
        logger.error(f"Missing bbox files for {brain_id}")
        return
    
    bbox_ref = load_bbox(bbox_ref_path, 'bbox')
    bbox_reg = load_bbox(bbox_reg_path, 'bbox_reg')
    
    # Find reference H5 to get pixel size
    ref_h5_pattern = f"*round00_{brain_id}_channel-DAPI.h5"
    ref_h5_files = list(h5_dir.glob(ref_h5_pattern))
    
    if len(ref_h5_files) == 0:
        logger.error(f"No reference H5 file found for {brain_id}")
        return
    
    ref_h5_path = ref_h5_files[0]
    
    # Get pixel size from H5 attributes
    try:
        with h5py.File(ref_h5_path, 'r') as f:
            pixel_size = f['/data'].attrs.get('element_size_um', [0.108, 0.108, 0.4])[0]
    except Exception as e:
        logger.error(f"Failed to load pixel size from {ref_h5_path}: {e}")
        raise
    
    logger.info(f"Using pixel_size={pixel_size}, ref_step={ref_step}")
    
    # Get cropped size from bbox_reg
    cropped_size = get_cropped_image_size(ref_h5_path, bbox_reg)
    
    # Find all displacement field files
    D_files = list(regdemon_dir.glob(f"*{cpd}_cropped_corr_D_regdemon.h5"))
    
    if len(D_files) == 0:
        logger.warning(f"No displacement files found for {brain_id}")
        return
    
    # Extract round IDs from filenames
    round_pattern = re.compile(r'round(\d+)')
    rounds = []
    
    for D_file in D_files:
        match = round_pattern.search(D_file.name)
        if match:
            rounds.append(match.group(0))  # e.g., "round01"
    
    logger.info(f"Found {len(rounds)} rounds: {sorted(rounds)}")
    
    # Process each round
    for round_id in sorted(rounds):
        logger.info(f"Processing {round_id}")
        
        # Find displacement field file
        D_pattern = f"*{round_id}*{cpd}_cropped_corr_D_regdemon.h5"
        D_files_round = list(regdemon_dir.glob(D_pattern))
        
        if len(D_files_round) == 0:
            logger.warning(f"No D file for {round_id}")
            continue
        
        D_h5_path = D_files_round[0]
        
        # Find corresponding affine transform
        T_pattern = f"*{round_id}*{cpd}_cropped_corr.h5"
        T_files = list(corr_dir.glob(T_pattern))
        
        if len(T_files) == 0:
            logger.warning(f"No T file for {round_id}")
            continue
        
        T_h5_path = T_files[0]
        
        # Load transformation and displacement field
        T_forward = load_affine_transform(T_h5_path)
        D = load_displacement_field(D_h5_path)
        
        # Resize displacement field to cropped size
        D_resized = resize_displacement_field(D, cropped_size)
        
        # Find puncta CSVs for this round (search recursively for subdirectory structure)
        puncta_pattern = f"*{round_id}*{brain_id}*.csv"
        puncta_files = list(Path(puncta_dir).rglob(puncta_pattern))
        
        if len(puncta_files) == 0:
            logger.warning(f"No puncta files for {round_id}")
            continue
        
        logger.info(f"Found {len(puncta_files)} puncta files for {round_id}")
        
        # Process each puncta file
        for puncta_csv in puncta_files:
            # Generate output filename
            stub = puncta_csv.stem
            output_path = output_dir / "pixel" / f"{stub}_regis.csv"
            
            # Register puncta
            result = register_puncta_for_round(
                puncta_csv,
                T_forward,
                D_resized,
                bbox_reg,
                pixel_size,
                ref_step,
                output_path
            )
            
            if result is None:
                logger.info(f"Skipped {puncta_csv.name} (no puncta remaining after registration)")
    
    logger.info(f"Completed registration for {brain_id}")
