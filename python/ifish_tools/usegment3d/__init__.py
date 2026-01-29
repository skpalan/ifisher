"""
u-segment3D pipeline module for 3D cell segmentation.

This module provides a generalized, reusable pipeline for running u-segment3D
on fly brain data or other 3D microscopy images.

Usage
-----
# From command line
python -m ifish_tools.usegment3d --config config.yaml

# From Python
from ifish_tools.usegment3d import USeg3DPipeline

pipeline = USeg3DPipeline.from_yaml('config.yaml')
results = pipeline.run()
"""

from .config import (
    BoundingBox,
    CloneConfig,
    BrainConfig,
    PipelineConfig,
    generate_config_template
)

from .pipeline import USeg3DPipeline

__all__ = [
    'BoundingBox',
    'CloneConfig',
    'BrainConfig',
    'PipelineConfig',
    'USeg3DPipeline',
    'generate_config_template'
]
