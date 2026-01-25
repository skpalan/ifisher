"""I/O utilities for loading and saving masks, transforms, and puncta data."""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import tifffile


def load_mask(path: Union[str, Path]) -> np.ndarray:
    """Load a 3D label mask from a TIFF file.

    Args:
        path: Path to the TIFF file.

    Returns:
        3D numpy array with shape (Z, Y, X) containing integer labels.
        Label 0 is background, labels 1-N are cells.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    mask = tifffile.imread(path)
    return mask


def save_mask(mask: np.ndarray, path: Union[str, Path], compress: bool = True) -> None:
    """Save a 3D label mask to a TIFF file.

    Args:
        mask: 3D numpy array with integer labels.
        path: Output path for the TIFF file.
        compress: Whether to use compression (default True).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    compression = "zlib" if compress else None
    tifffile.imwrite(path, mask, compression=compression)


def save_transform(transform_params: dict, path: Union[str, Path]) -> None:
    """Save transformation parameters to a JSON file.

    The transform_params dict should contain:
        - anchors_original: List of original anchor points [[z,y,x], ...]
        - anchors_transformed: List of transformed anchor points
        - cells: Dict mapping cell_id to cell info including:
            - centroid_original: [z, y, x]
            - centroid_transformed: [z', y', x']
            - anchor_index: int
            - phi: rotation angle
            - theta: rotation angle

    Args:
        transform_params: Dictionary containing transformation parameters.
        path: Output path for the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_params = convert_to_serializable(transform_params)

    with open(path, "w") as f:
        json.dump(serializable_params, f, indent=2)


def load_transform(path: Union[str, Path]) -> dict:
    """Load transformation parameters from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary containing transformation parameters with numpy arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")

    with open(path, "r") as f:
        params = json.load(f)

    # Convert lists back to numpy arrays for key fields
    if "anchors_original" in params:
        params["anchors_original"] = np.array(params["anchors_original"])
    if "anchors_transformed" in params:
        params["anchors_transformed"] = np.array(params["anchors_transformed"])

    # Convert cell centroids to numpy arrays
    if "cells" in params:
        for cell_id, cell_info in params["cells"].items():
            if "centroid_original" in cell_info:
                cell_info["centroid_original"] = np.array(cell_info["centroid_original"])
            if "centroid_transformed" in cell_info:
                cell_info["centroid_transformed"] = np.array(
                    cell_info["centroid_transformed"]
                )

    return params


def load_puncta(path: Union[str, Path]) -> pd.DataFrame:
    """Load puncta locations from a CSV file.

    Expects columns: x, y, z (and optionally others like intensity, cell_id).
    Note: The returned DataFrame uses the original column order from the file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with puncta data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Puncta file not found: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    required_cols = {"x", "y", "z"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Puncta file must contain columns {required_cols}, "
            f"found: {set(df.columns)}"
        )

    return df


def save_puncta(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save puncta locations to a CSV file.

    Args:
        df: DataFrame with puncta data.
        path: Output path for the CSV file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
