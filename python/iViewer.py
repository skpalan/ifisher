#!/usr/bin/env python
"""
cycleHCR 3D Visualization with napari

Interactive exploration of spatial transcriptomic data:
- Reference DAPI and GFP channels
- Cell segmentation with 3D bounding boxes
- Registered puncta images from multiple cycles
- Detected puncta centroid locations

Usage:
    python iViewer.py --dir <path_to_data_dir>

The data directory should contain three subdirectories:
    - segmentation/: Reference images and cell masks
    - regis_puncta_img/: Registered puncta images
    - regis_puncta_loc/pixel/: Puncta location CSVs
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread
import napari
from napari_orthogonal_views.ortho_view_manager import show_orthogonal_views

# Enable async rendering for better GPU performance
os.environ["NAPARI_ASYNC"] = "1"

# Channel color mapping
CHANNEL_COLORS = {"Cy3": "yellow", "Cy5": "red", "mCherry": "magenta"}


def get_z_dim(img):
    """Get z-dimension from image array."""
    return img.shape[0]  # Works for both (Z,Y,X) and (Z,C,Y,X)


def load_reference_image(ref_path):
    """Load and split a reference image into DAPI and GFP channels."""
    ref_img = imread(ref_path)
    print(f"Reference image: {ref_path.name}")
    print(f"Reference image shape: {ref_img.shape}")
    
    ref_z = get_z_dim(ref_img)
    print(f"Reference Z slices: {ref_z}")
    
    # Split channels (assuming Z, C, Y, X or similar)
    if ref_img.ndim == 4:
        if ref_img.shape[1] <= 4:  # (Z, C, Y, X)
            dapi = ref_img[:, 0, :, :]
            gfp = ref_img[:, 1, :, :] if ref_img.shape[1] > 1 else None
        else:  # (C, Z, Y, X)
            dapi = ref_img[0]
            gfp = ref_img[1] if ref_img.shape[0] > 1 else None
    else:
        dapi = ref_img
        gfp = None
    
    print(f"DAPI shape: {dapi.shape}")
    if gfp is not None:
        print(f"GFP shape: {gfp.shape}")
    
    return dapi, gfp, ref_z


def load_reference_and_masks(seg_dir, ref_pattern=None):
    """Load reference image channels and cell masks from segmentation directory.
    
    Args:
        seg_dir: Path to segmentation directory
        ref_pattern: Optional filename pattern to search for reference image.
                     If None, searches for *_full_cropped.tif files.
    """
    # Find reference image
    if ref_pattern:
        # Search for files matching the pattern
        ref_files = list(seg_dir.glob(f"*{ref_pattern}*"))
        ref_files = [f for f in ref_files if f.suffix.lower() in ('.tif', '.tiff')]
        if not ref_files:
            raise FileNotFoundError(f"No reference image matching '{ref_pattern}' found in {seg_dir}")
    else:
        # Default: find *_full_cropped.tif files
        ref_files = list(seg_dir.glob("*_full_cropped.tif"))
        ref_files = [f for f in ref_files if "cp_masks" not in f.name]
        if not ref_files:
            raise FileNotFoundError(f"No reference image found in {seg_dir}")
    
    ref_path = ref_files[0]
    dapi, gfp, ref_z = load_reference_image(ref_path)
    
    # Find and load cell masks (*cp_masks*.tif)
    mask_files = list(seg_dir.glob("*cp_masks*.tif"))
    masks = None
    if mask_files:
        mask_path = mask_files[0]
        masks = imread(mask_path)
        print(f"Cell masks: {mask_path.name}")
        print(f"Cell masks shape: {masks.shape}, cells: {len(np.unique(masks)) - 1}")
    else:
        print("No cell masks found")
    
    return dapi, gfp, masks, ref_z


def add_puncta_images(viewer, puncta_img_dir, ref_z):
    """Load and add puncta images with dynamic z-scaling."""
    puncta_z_scales = {}
    
    tif_files = sorted(puncta_img_dir.glob("*.tif"))
    if not tif_files:
        print(f"No puncta images found in {puncta_img_dir}")
        return puncta_z_scales
    
    print(f"\nLoading {len(tif_files)} puncta images...")
    
    for tif_path in tif_files:
        name = tif_path.stem
        
        # Parse round and channel from filename
        try:
            round_id = name.split("_")[1]  # e.g., "round01"
            channel = name.split("channel-")[1].split("_")[0]
        except (IndexError, ValueError):
            print(f"  Skipping {name}: could not parse filename")
            continue
        
        channel_base = channel.replace(" Nar", "")
        color = CHANNEL_COLORS.get(channel_base, "white")
        
        img = imread(tif_path)
        
        # Calculate z-scale dynamically
        puncta_z = get_z_dim(img)
        z_scale = ref_z / puncta_z
        puncta_z_scales[f"{round_id}_{channel}"] = z_scale
        
        viewer.add_image(
            img,
            name=f"Puncta {round_id} {channel}",
            colormap=color,
            blending="additive",
            visible=False,  # Start hidden to avoid clutter
            scale=(z_scale, 1, 1),  # Scale z-axis to match reference
        )
        print(f"  Added image: {round_id} {channel} (z_scale={z_scale:.2f})")
    
    return puncta_z_scales


def add_puncta_locations(viewer, puncta_loc_dir, ref_z, puncta_z_scales):
    """Load and add puncta location points with dynamic z-scaling."""
    csv_files = sorted(puncta_loc_dir.glob("*.csv"))
    if not csv_files:
        print(f"No puncta location CSVs found in {puncta_loc_dir}")
        return
    
    print(f"\nLoading {len(csv_files)} puncta location files...")
    
    for csv_path in csv_files:
        name = csv_path.stem
        parts = name.split("_")
        
        # Parse round and channel from filename
        try:
            round_id = parts[3]  # e.g., "round01"
            channel = name.split("channel-")[1].split("_")[0]
        except (IndexError, ValueError):
            print(f"  Skipping {name}: could not parse filename")
            continue
        
        channel_base = channel.replace(" Nar", "")
        color = CHANNEL_COLORS.get(channel_base, "white")
        
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            coords = df[["z", "y", "x"]].values.copy()  # napari expects (z, y, x)
            
            # Get z-scale from corresponding image, or use default
            key = f"{round_id}_{channel}"
            z_scale = puncta_z_scales.get(key, ref_z / 251)  # fallback to common ratio
            
            # Scale z-coordinates to match reference
            coords[:, 0] = coords[:, 0] * z_scale
            
            viewer.add_points(
                coords,
                name=f"Loc {round_id} {channel}",
                face_color=color,
                size=3,
                blending="translucent",
                visible=False,
            )
            print(f"  Added points: {round_id} {channel} ({len(df)} pts, z_scale={z_scale:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="cycleHCR 3D Visualization with napari",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The data directory should contain three subdirectories:
  - segmentation/: Reference images and cell masks
  - regis_puncta_img/: Registered puncta images  
  - regis_puncta_loc/pixel/: Puncta location CSVs
        """,
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=False,
        default=None,
        help="Path to data directory containing segmentation, regis_puncta_img, and regis_puncta_loc subdirectories. If not provided, starts an empty viewer.",
    )
    parser.add_argument(
        "--ortho",
        action="store_true",
        default=False,
        help="Enable orthogonal views plugin (default: disabled)",
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=False,
        default=None,
        help="Reference image. When --dir is specified, this is a filename pattern to search in the segmentation subfolder. When --dir is not specified, this is a full path to a reference image.",
    )
    args = parser.parse_args()
    
    # If no directory provided
    if args.dir is None:
        viewer = napari.Viewer(title="cycleHCR 3D")
        
        # Load reference image if provided
        if args.ref is not None:
            ref_path = Path(args.ref).resolve()
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference image not found: {ref_path}")
            
            print(f"Loading reference image: {ref_path}")
            dapi, gfp, ref_z = load_reference_image(ref_path)
            
            viewer.add_image(dapi, name="DAPI", colormap="blue", blending="additive")
            if gfp is not None:
                viewer.add_image(gfp, name="GFP membrane", colormap="green", blending="additive")
            
            viewer.dims.ndisplay = 3
            viewer.camera.angles = (2, 15, 150)
        else:
            print("No data directory provided. Starting empty viewer...")
        
        if args.ortho:
            show_orthogonal_views(viewer)
        napari.run()
        return
    
    # Set up data directories
    data_dir = Path(args.dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    seg_dir = data_dir / "segmentation"
    puncta_img_dir = data_dir / "regis_puncta_img"
    puncta_loc_dir = data_dir / "regis_puncta_loc" / "pixel"
    
    # Validate directories exist
    for dir_path, name in [(seg_dir, "segmentation"), (puncta_img_dir, "regis_puncta_img")]:
        if not dir_path.exists():
            raise FileNotFoundError(f"{name} directory not found: {dir_path}")
    
    print(f"Data directory: {data_dir}")
    print(f"Segmentation: {seg_dir}")
    print(f"Puncta images: {puncta_img_dir}")
    print(f"Puncta locations: {puncta_loc_dir}")
    print()
    
    # Load reference channels and cell masks
    print("=" * 50)
    print("Loading reference channels and cell masks...")
    print("=" * 50)
    dapi, gfp, masks, ref_z = load_reference_and_masks(seg_dir, ref_pattern=args.ref)
    
    # Create viewer
    print()
    print("=" * 50)
    print("Creating napari viewer...")
    print("=" * 50)
    viewer = napari.Viewer(title="cycleHCR 3D")
    
    # Add reference channels
    viewer.add_image(dapi, name="DAPI", colormap="blue", blending="additive")
    if gfp is not None:
        viewer.add_image(gfp, name="GFP membrane", colormap="green", blending="additive")
    
    # Add cell masks with 3D bounding box overlay
    if masks is not None:
        labels_layer = viewer.add_labels(masks, name="Cell masks", opacity=0.5)
        labels_layer.bounding_box.visible = True
        labels_layer.bounding_box.line_color = "cyan"
    
    # Add puncta images
    print()
    print("=" * 50)
    print("Loading puncta images...")
    print("=" * 50)
    puncta_z_scales = add_puncta_images(viewer, puncta_img_dir, ref_z)
    
    # Add puncta locations
    print()
    print("=" * 50)
    print("Loading puncta locations...")
    print("=" * 50)
    if puncta_loc_dir.exists():
        add_puncta_locations(viewer, puncta_loc_dir, ref_z, puncta_z_scales)
    else:
        print(f"Puncta location directory not found: {puncta_loc_dir}")
    
    # Switch to 3D view and set camera angle
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (2, 15, 150)
    
    # Enable orthogonal views plugin if requested
    if args.ortho:
        show_orthogonal_views(viewer)
    
    print()
    print("=" * 50)
    print("Done! Toggle layer visibility in the layer list panel.")
    print("Press '3' to switch between 2D/3D view")
    if args.ortho:
        print("Press 'T' to center all orthogonal views to current mouse location")
    print("=" * 50)
    
    # Start the napari event loop
    napari.run()


if __name__ == "__main__":
    main()
