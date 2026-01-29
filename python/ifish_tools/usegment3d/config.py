"""
Configuration handling for u-segment3D pipeline.

Defines dataclasses for pipeline configuration and YAML parsing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")


@dataclass
class BoundingBox:
    """Bounding box for a clone region in Y, X, Z format."""
    
    y: tuple[int, int]  # (ymin, ymax)
    x: tuple[int, int]  # (xmin, xmax)
    z: tuple[int, int]  # (zmin, zmax)
    
    def to_slices(self) -> tuple[slice, slice, slice]:
        """
        Return numpy slices in (Z, Y, X) order for indexing.
        
        Returns
        -------
        tuple[slice, slice, slice]
            (z_slice, y_slice, x_slice)
        """
        return (
            slice(self.z[0], self.z[1]),
            slice(self.y[0], self.y[1]),
            slice(self.x[0], self.x[1])
        )
    
    def shape(self) -> tuple[int, int, int]:
        """
        Return shape as (Z, Y, X).
        
        Returns
        -------
        tuple[int, int, int]
            (nz, ny, nx)
        """
        return (
            self.z[1] - self.z[0],
            self.y[1] - self.y[0],
            self.x[1] - self.x[0]
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BoundingBox':
        """Create BoundingBox from dictionary."""
        return cls(
            y=tuple(data['y']),
            x=tuple(data['x']),
            z=tuple(data['z'])
        )


@dataclass
class CloneConfig:
    """Configuration for a single clone region."""
    
    name: str
    bbox: BoundingBox
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CloneConfig':
        """Create CloneConfig from dictionary."""
        return cls(
            name=data['name'],
            bbox=BoundingBox.from_dict(data['bbox'])
        )


@dataclass
class BrainConfig:
    """Configuration for a brain with one or more clones."""
    
    name: str
    path: str
    channel: int
    clones: list[CloneConfig]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BrainConfig':
        """Create BrainConfig from dictionary."""
        clones = [CloneConfig.from_dict(c) for c in data['clones']]
        return cls(
            name=data['name'],
            path=data['path'],
            channel=data['channel'],
            clones=clones
        )
    
    def validate(self) -> list[str]:
        """
        Validate brain configuration.
        
        Returns
        -------
        list[str]
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check file exists
        if not os.path.exists(self.path):
            errors.append(f"Brain {self.name}: file not found: {self.path}")
        
        # Check channel is non-negative
        if self.channel < 0:
            errors.append(f"Brain {self.name}: channel must be >= 0, got {self.channel}")
        
        # Check at least one clone
        if not self.clones:
            errors.append(f"Brain {self.name}: must have at least one clone")
        
        # Check clone bounding boxes are valid
        for clone in self.clones:
            bbox = clone.bbox
            if bbox.y[0] >= bbox.y[1]:
                errors.append(f"Brain {self.name}, clone {clone.name}: invalid Y range {bbox.y}")
            if bbox.x[0] >= bbox.x[1]:
                errors.append(f"Brain {self.name}, clone {clone.name}: invalid X range {bbox.x}")
            if bbox.z[0] >= bbox.z[1]:
                errors.append(f"Brain {self.name}, clone {clone.name}: invalid Z range {bbox.z}")
        
        return errors


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    
    # Metadata
    pipeline_name: str = "usegment3d_pipeline"
    pipeline_date: str = "0128"
    pipeline_description: str = ""
    
    # Cellpose settings
    cellpose_model_path: str = ""
    cellpose_diameter: Optional[float] = None
    cellpose_channels: tuple[int, int] = (0, 0)
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    
    # u-segment3D aggregation settings
    aggregation_method: str = "direct"
    aggregation_remove_small_objects: int = 1000
    aggregation_min_area: int = 500
    aggregation_min_prob_thresh: float = 0.25
    aggregation_gradient_decay: float = 0.0
    aggregation_n_iter: int = 200
    
    # Smoothing settings
    smoothing_enabled: bool = True
    smoothing_method: str = "label_diffusion"
    smoothing_refine_iters: int = 15
    smoothing_refine_alpha: float = 0.60
    smoothing_refine_clamp: float = 0.7
    
    # Output settings
    output_directory: str = ""
    output_naming_template: str = "{brain_base}_useg_{date}_cp_masks.tif"
    output_save_raw: bool = True
    output_save_smoothed: bool = True
    output_save_visualization: bool = False
    output_visualization_slices: list[int] = field(default_factory=lambda: [110, 120, 130, 140])
    
    # Processing settings
    processing_gpus: list[int] = field(default_factory=lambda: [0])
    processing_parallel: bool = False
    processing_skip_existing: bool = True
    processing_dry_run: bool = False
    
    # Brain data
    brains: list[BrainConfig] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        path : str
            Path to YAML configuration file
            
        Returns
        -------
        PipelineConfig
            Loaded configuration
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse pipeline metadata
        pipeline = data.get('pipeline', {})
        
        # Parse cellpose settings
        cellpose = data.get('cellpose', {})
        
        # Parse aggregation settings
        aggregation = data.get('aggregation', {})
        
        # Parse smoothing settings
        smoothing = data.get('smoothing', {})
        smoothing_params = smoothing.get('params', {})
        
        # Parse output settings
        output = data.get('output', {})
        
        # Parse processing settings
        processing = data.get('processing', {})
        
        # Parse brains
        brains = [BrainConfig.from_dict(b) for b in data.get('brains', [])]
        
        return cls(
            # Metadata
            pipeline_name=pipeline.get('name', 'usegment3d_pipeline'),
            pipeline_date=pipeline.get('date', '0128'),
            pipeline_description=pipeline.get('description', ''),
            
            # Cellpose
            cellpose_model_path=cellpose.get('model_path', ''),
            cellpose_diameter=cellpose.get('diameter'),
            cellpose_channels=tuple(cellpose.get('channels', [0, 0])),
            cellpose_flow_threshold=cellpose.get('flow_threshold', 0.4),
            cellpose_cellprob_threshold=cellpose.get('cellprob_threshold', 0.0),
            
            # Aggregation
            aggregation_method=aggregation.get('method', 'direct'),
            aggregation_remove_small_objects=aggregation.get('remove_small_objects', 1000),
            aggregation_min_area=aggregation.get('min_area', 500),
            aggregation_min_prob_thresh=aggregation.get('min_prob_thresh', 0.25),
            aggregation_gradient_decay=aggregation.get('gradient_decay', 0.0),
            aggregation_n_iter=aggregation.get('n_iter', 200),
            
            # Smoothing
            smoothing_enabled=smoothing.get('enabled', True),
            smoothing_method=smoothing.get('method', 'label_diffusion'),
            smoothing_refine_iters=smoothing_params.get('refine_iters', 15),
            smoothing_refine_alpha=smoothing_params.get('refine_alpha', 0.60),
            smoothing_refine_clamp=smoothing_params.get('refine_clamp', 0.7),
            
            # Output
            output_directory=output.get('directory', ''),
            output_naming_template=output.get('naming_template', '{brain_base}_useg_{date}_cp_masks.tif'),
            output_save_raw=output.get('save_raw', True),
            output_save_smoothed=output.get('save_smoothed', True),
            output_save_visualization=output.get('save_visualization', False),
            output_visualization_slices=output.get('visualization_slices', [110, 120, 130, 140]),
            
            # Processing
            processing_gpus=processing.get('gpus', [0]),
            processing_parallel=processing.get('parallel', False),
            processing_skip_existing=processing.get('skip_existing', True),
            processing_dry_run=processing.get('dry_run', False),
            
            # Brains
            brains=brains
        )
    
    def validate(self) -> list[str]:
        """
        Validate entire pipeline configuration.
        
        Returns
        -------
        list[str]
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check cellpose model exists
        if not self.cellpose_model_path:
            errors.append("Cellpose model path is required")
        elif not os.path.exists(self.cellpose_model_path):
            errors.append(f"Cellpose model not found: {self.cellpose_model_path}")
        
        # Check output directory
        if not self.output_directory:
            errors.append("Output directory is required")
        
        # Check at least one brain
        if not self.brains:
            errors.append("At least one brain must be configured")
        
        # Check aggregation method
        if self.aggregation_method not in ['direct', 'indirect']:
            errors.append(f"Invalid aggregation method: {self.aggregation_method}")
        
        # Check smoothing method
        if self.smoothing_enabled and self.smoothing_method not in ['label_diffusion']:
            errors.append(f"Invalid smoothing method: {self.smoothing_method}")
        
        # Validate each brain
        for brain in self.brains:
            errors.extend(brain.validate())
        
        return errors
    
    def get_output_path(self, brain_name: str, raw: bool = False) -> str:
        """
        Get output file path for a brain.
        
        Parameters
        ----------
        brain_name : str
            Brain name (e.g., 'brain08')
        raw : bool
            If True, return path for raw (unsmoothed) output
            
        Returns
        -------
        str
            Full output file path
        """
        # Extract base name from brain path if needed
        # Assume brain_name is like "brain08" and we want "Gel20251024_round00_brain08_intact_cropped"
        for brain in self.brains:
            if brain.name == brain_name:
                brain_base = Path(brain.path).stem  # e.g., "Gel20251024_round00_brain08_intact_cropped"
                break
        else:
            brain_base = brain_name
        
        # Generate filename
        if raw:
            template = self.output_naming_template.replace('.tif', '_raw.tif')
        else:
            template = self.output_naming_template
        
        filename = template.format(
            brain_base=brain_base,
            brain_name=brain_name,
            date=self.pipeline_date
        )
        
        return os.path.join(self.output_directory, filename)
    
    def get_clone_output_path(self, brain_name: str, clone_name: str, raw: bool = False) -> str:
        """
        Get output path for a specific clone.
        
        Parameters
        ----------
        brain_name : str
            Name of the brain (e.g., 'brain08')
        clone_name : str
            Name of the clone (e.g., 'clone1')
        raw : bool
            If True, return path for raw (unsmoothed) output
            
        Returns
        -------
        str
            Full output file path
        """
        brain_config = next((b for b in self.brains if b.name == brain_name), None)
        if brain_config is None:
            raise ValueError(f"Brain {brain_name} not found in config")
        
        brain_base = Path(brain_config.path).stem
        suffix = "_raw" if raw else ""
        filename = f"{brain_base}_{clone_name}_useg_{self.pipeline_date}{suffix}_cp_masks.tif"
        return os.path.join(self.output_directory, filename)


def generate_config_template(output_path: str) -> None:
    """
    Generate a template YAML configuration file.
    
    Parameters
    ----------
    output_path : str
        Where to save the template file
    """
    template = """# u-segment3D Pipeline Configuration

# Pipeline metadata
pipeline:
  name: "fly_brain_segmentation"
  date: "0128"
  description: "3D cell segmentation of fly brain clones"

# Cellpose settings
cellpose:
  model_path: "/path/to/cellpose/model"
  diameter: null          # null = auto-detect
  channels: [0, 0]        # [cyto_channel, nucleus_channel], [0,0] = grayscale
  flow_threshold: 0.4
  cellprob_threshold: 0.0

# u-segment3D aggregation settings
aggregation:
  method: "direct"        # "direct" or "indirect"
  remove_small_objects: 1000
  min_area: 500
  min_prob_thresh: 0.25
  gradient_decay: 0.0     # DO NOT increase (destroys cells)
  n_iter: 200

# Smoothing settings
smoothing:
  enabled: true
  method: "label_diffusion"
  params:
    refine_iters: 15      # Number of diffusion iterations
    refine_alpha: 0.60    # Image-guided weight (0-1)
    refine_clamp: 0.7     # Clamping value

# Output settings
output:
  directory: "/path/to/output"
  naming_template: "{brain_base}_useg_{date}_cp_masks.tif"
  save_raw: true          # Save pre-smoothing result
  save_smoothed: true     # Save post-smoothing result
  save_visualization: false
  visualization_slices: [110, 120, 130, 140]

# Processing settings
processing:
  gpus: [0, 1]            # GPU IDs for parallel processing
  parallel: true
  skip_existing: true     # Skip already processed brains
  dry_run: false          # Validate without processing

# Brain data
brains:
  - name: "brain08"
    path: "/path/to/brain08.tif"
    channel: 0            # DAPI channel index
    clones:
      - name: "clone1"
        bbox:
          y: [560, 788]   # [ymin, ymax]
          x: [775, 1210]  # [xmin, xmax]
          z: [71, 420]    # [zmin, zmax]

  - name: "brain11"
    path: "/path/to/brain11.tif"
    channel: 0
    clones:
      - name: "clone1"
        bbox: {y: [396, 646], x: [406, 687], z: [40, 348]}
      - name: "clone2"
        bbox: {y: [340, 662], x: [1068, 1376], z: [184, 480]}
"""
    
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"Template configuration written to: {output_path}")
