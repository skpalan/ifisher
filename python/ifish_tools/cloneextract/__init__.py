"""Clone mask extraction with morphological closing."""

from .config import BoxCoords, BrainCloneSpec, CloneExtractConfig, load_bbox_from_mat
from .core import (
    detect_mask_space,
    compute_relative_cbox,
    apply_morphological_closing,
    extract_clone_mask,
    process_brain,
    run_pipeline,
)

__all__ = [
    "BoxCoords", "BrainCloneSpec", "CloneExtractConfig", "load_bbox_from_mat",
    "detect_mask_space", "compute_relative_cbox", "apply_morphological_closing",
    "extract_clone_mask", "process_brain", "run_pipeline",
]
