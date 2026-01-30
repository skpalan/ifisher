"""Configuration for clone mask extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BoxCoords:
    """Bounding box coordinates in 0-indexed Python convention.

    Internal storage: ymin, ymax, xmin, xmax, zmin, zmax (all 0-indexed).
    ymax, xmax, zmax are exclusive (Python slice convention).
    """

    ymin: int
    ymax: int
    xmin: int
    xmax: int
    zmin: int
    zmax: int

    @classmethod
    def from_matlab(cls, coords: list[int]) -> "BoxCoords":
        """Create from MATLAB 1-indexed [ymin, ymax, xmin, xmax, zmin, zmax].

        Converts start indices by subtracting 1 (MATLAB 1-indexed → Python 0-indexed).
        End indices stay the same (MATLAB inclusive end → Python exclusive end).
        """
        ymin, ymax, xmin, xmax, zmin, zmax = coords
        return cls(
            ymin=ymin - 1, ymax=ymax,
            xmin=xmin - 1, xmax=xmax,
            zmin=zmin - 1, zmax=zmax,
        )

    @classmethod
    def from_python(cls, coords: list[int]) -> "BoxCoords":
        """Create from 0-indexed [ymin, ymax, xmin, xmax, zmin, zmax]."""
        ymin, ymax, xmin, xmax, zmin, zmax = coords
        return cls(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax)

    def shape(self) -> tuple[int, int, int]:
        """Return (Z, Y, X) shape."""
        return (self.zmax - self.zmin, self.ymax - self.ymin, self.xmax - self.xmin)

    def to_slices(self) -> tuple[slice, slice, slice]:
        """Return (z_slice, y_slice, x_slice) for numpy indexing."""
        return (
            slice(self.zmin, self.zmax),
            slice(self.ymin, self.ymax),
            slice(self.xmin, self.xmax),
        )


@dataclass
class BrainCloneSpec:
    """Specification for one brain's clones."""

    brain_name: str       # e.g. "brain08"
    mask_path: str        # path to mask TIFF
    bbox: BoxCoords       # B-box (cropping bounding box)
    clones: dict[str, BoxCoords]  # clone_name → C-box


@dataclass
class CloneExtractConfig:
    """Full configuration for clone extraction pipeline."""

    brains: list[BrainCloneSpec]
    output_dir: str
    closing_radius: int = 5
    date_tag: str = "0129"
    naming_template: str = "{brain_base}_{clone_name}_useg_{date}_cp_masks.tif"

    @classmethod
    def from_yaml(cls, path: str) -> "CloneExtractConfig":
        """Load from YAML config file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        brains = []
        for b in data["brains"]:
            bbox_data = b["bbox"]
            if isinstance(bbox_data, str):
                bbox = load_bbox_from_mat(bbox_data)
            else:
                bbox = BoxCoords.from_matlab(bbox_data)

            clones = {}
            for cname, ccoords in b["clones"].items():
                clones[cname] = BoxCoords.from_matlab(ccoords)

            brains.append(BrainCloneSpec(
                brain_name=b["name"],
                mask_path=b["mask_path"],
                bbox=bbox,
                clones=clones,
            ))

        return cls(
            brains=brains,
            output_dir=data.get("output_dir", "."),
            closing_radius=data.get("closing_radius", 5),
            date_tag=data.get("date_tag", "0129"),
            naming_template=data.get("naming_template",
                                     "{brain_base}_{clone_name}_useg_{date}_cp_masks.tif"),
        )


def load_bbox_from_mat(mat_path: str) -> BoxCoords:
    """Load B-box from a bbox_ref.mat file.

    The .mat file has 'bbox' key with nested structure
    holding [ymin, ymax, xmin, xmax, zmin, zmax] as 1-indexed MATLAB values.
    """
    import scipy.io

    data = scipy.io.loadmat(mat_path)
    bbox_raw = data["bbox"]
    coords = [int(bbox_raw[0, 0][i][0, 0]) for i in range(6)]
    return BoxCoords.from_matlab(coords)
