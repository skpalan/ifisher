"""
Main pipeline orchestration for u-segment3D processing.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from cellpose import models

from .config import PipelineConfig, BrainConfig, BoundingBox
from .cellpose_views import run_cellpose_3_views
from .aggregation import aggregate_direct_method
from .smoothing import apply_label_diffusion
from .utils import (
    combine_clone_masks,
    save_segmentation_mask,
    create_visualization,
    check_existing_outputs,
    save_processing_log
)


class USeg3DPipeline:
    """
    u-segment3D pipeline for 3D cell segmentation from 2D orthogonal views.
    
    Usage
    -----
    # From config file
    pipeline = USeg3DPipeline.from_yaml('config.yaml')
    results = pipeline.run()
    
    # With overrides
    pipeline.run(dry_run=True, gpus=[0, 1])
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.
        
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        """
        self.config = config
        self.results = {}
        self.cellpose_model = None
        
    @classmethod
    def from_yaml(cls, path: str) -> 'USeg3DPipeline':
        """
        Load pipeline from YAML configuration file.
        
        Parameters
        ----------
        path : str
            Path to YAML config file
            
        Returns
        -------
        USeg3DPipeline
            Initialized pipeline
        """
        config = PipelineConfig.from_yaml(path)
        return cls(config)
    
    def validate_config(self) -> None:
        """
        Validate configuration and raise errors if invalid.
        
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        errors = self.config.validate()
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
    
    def load_cellpose_model(self, gpu_id: int = 0) -> models.CellposeModel:
        """
        Load Cellpose model.
        
        Parameters
        ----------
        gpu_id : int
            GPU ID to use
            
        Returns
        -------
        CellposeModel
            Loaded model
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        model = models.CellposeModel(
            gpu=True,
            pretrained_model=self.config.cellpose_model_path
        )
        return model
    
    def run(self,
            dry_run: bool = None,
            skip_existing: bool = None,
            gpus: list = None,
            brains: list = None) -> dict:
        """
        Run the pipeline on all configured brains.
        
        Parameters
        ----------
        dry_run : bool, optional
            If True, validate config and show what would be processed
        skip_existing : bool, optional
            If True, skip brains with existing output files
        gpus : list, optional
            GPU IDs to use for parallel processing
        brains : list, optional
            List of brain names to process (None = all)
            
        Returns
        -------
        dict
            Results with cell counts, timing, and file paths
        """
        # Override config settings if provided
        if dry_run is not None:
            self.config.processing_dry_run = dry_run
        if skip_existing is not None:
            self.config.processing_skip_existing = skip_existing
        if gpus is not None:
            self.config.processing_gpus = gpus
        
        # Validate configuration
        print("="*70)
        print("u-segment3D Pipeline")
        print("="*70)
        print(f"Config: {self.config.pipeline_name} (date: {self.config.pipeline_date})")
        print(f"Output: {self.config.output_directory}")
        print(f"Brains: {len(self.config.brains)}")
        print(f"GPUs: {self.config.processing_gpus}")
        print(f"Parallel: {self.config.processing_parallel}")
        print(f"Dry run: {self.config.processing_dry_run}")
        print()
        
        try:
            self.validate_config()
            print("✓ Configuration valid")
        except ValueError as e:
            print(f"✗ {e}")
            return {}
        
        # Filter brains if specified
        brains_to_process = self.config.brains
        if brains is not None:
            brain_names_set = set(brains)
            brains_to_process = [b for b in self.config.brains if b.name in brain_names_set]
            print(f"Processing subset: {[b.name for b in brains_to_process]}")
        
        # Dry run: just show what would be processed
        if self.config.processing_dry_run:
            print("\n" + "="*70)
            print("DRY RUN - Would process:")
            print("="*70)
            for brain in brains_to_process:
                print(f"\n{brain.name}:")
                print(f"  Path: {brain.path}")
                print(f"  Clones: {len(brain.clones)}")
                for clone in brain.clones:
                    print(f"    - {clone.name}: {clone.bbox.shape()}")
                
                # Check if outputs exist
                exists, existing = check_existing_outputs(
                    self.config.output_directory,
                    brain.name,
                    brain.path,
                    self.config.pipeline_date,
                    self.config.output_save_raw,
                    self.config.output_save_smoothed
                )
                if exists:
                    print(f"  Status: SKIP (outputs exist)")
                    for f in existing:
                        print(f"    {Path(f).name}")
                else:
                    print(f"  Status: PROCESS")
            
            print("\n" + "="*70)
            print("Dry run complete. No files were modified.")
            print("="*70)
            return {}
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        # Process brains
        start_time = time.time()
        
        if self.config.processing_parallel and len(self.config.processing_gpus) > 1:
            # Parallel processing
            results = self._process_parallel(brains_to_process)
        else:
            # Sequential processing
            results = self._process_sequential(brains_to_process)
        
        total_time = time.time() - start_time
        
        # Save processing log
        log_data = {
            'pipeline': {
                'name': self.config.pipeline_name,
                'date': self.config.pipeline_date,
                'total_time_seconds': total_time
            },
            'brains': results
        }
        
        log_path = os.path.join(self.config.output_directory, 'processing_log.json')
        save_processing_log(log_data, log_path)
        
        # Print summary
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Brains processed: {len(results)}")
        for brain_name, brain_result in results.items():
            if 'error' in brain_result:
                print(f"  {brain_name}: ERROR - {brain_result['error']}")
            else:
                total_cells = brain_result.get('total_cells_smoothed', brain_result.get('total_cells_raw', '?'))
                print(f"  {brain_name}: {total_cells} cells")
        print("="*70)
        
        return results
    
    def _process_sequential(self, brains: list) -> dict:
        """Process brains sequentially."""
        results = {}
        gpu_id = self.config.processing_gpus[0]
        
        for i, brain in enumerate(brains):
            print(f"\n[{i+1}/{len(brains)}] Processing {brain.name}...")
            try:
                result = self.process_brain(brain, gpu_id=gpu_id)
                results[brain.name] = result
            except Exception as e:
                print(f"  ERROR: {e}")
                results[brain.name] = {'error': str(e)}
        
        return results
    
    def _process_parallel(self, brains: list) -> dict:
        """Process brains in parallel using multiple GPUs."""
        results = {}
        gpus = self.config.processing_gpus
        
        with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
            # Submit jobs
            future_to_brain = {}
            for i, brain in enumerate(brains):
                gpu_id = gpus[i % len(gpus)]
                future = executor.submit(self.process_brain, brain, gpu_id)
                future_to_brain[future] = brain.name
            
            # Collect results
            for future in as_completed(future_to_brain):
                brain_name = future_to_brain[future]
                try:
                    result = future.result()
                    results[brain_name] = result
                    print(f"✓ {brain_name} complete")
                except Exception as e:
                    print(f"✗ {brain_name} failed: {e}")
                    results[brain_name] = {'error': str(e)}
        
        return results
    
    def process_brain(self, brain: BrainConfig, gpu_id: int = 0) -> dict:
        """
        Process a single brain (all its clones).
        
        Parameters
        ----------
        brain : BrainConfig
            Brain configuration
        gpu_id : int
            GPU ID to use
            
        Returns
        -------
        dict
            Processing results
        """
        start_time = time.time()
        
        # Check if outputs already exist
        if self.config.processing_skip_existing:
            exists, existing = check_existing_outputs(
                self.config.output_directory,
                brain.name,
                brain.path,
                self.config.pipeline_date,
                self.config.output_save_raw,
                self.config.output_save_smoothed
            )
            if exists:
                print(f"  Skipping {brain.name} (outputs exist)")
                return {'status': 'skipped', 'files': existing}
        
        # Load brain image
        print(f"  Loading brain from {brain.path}")
        brain_img = tifffile.imread(brain.path)
        print(f"    Shape: {brain_img.shape}")
        
        # Load Cellpose model
        if self.cellpose_model is None:
            print(f"  Loading Cellpose model (GPU {gpu_id})")
            self.cellpose_model = self.load_cellpose_model(gpu_id)
        
        # Process each clone
        clone_results = []
        clone_masks_raw = []
        clone_masks_smoothed = []
        
        for clone in brain.clones:
            print(f"\n  Processing {clone.name} (bbox: {clone.bbox.shape()})...")
            
            # Extract clone region (keep all channels)
            z_slice, y_slice, x_slice = clone.bbox.to_slices()
            clone_img = brain_img[z_slice, :, y_slice, x_slice].transpose(0, 2, 3, 1)  # (Z, C, Y, X) -> (Z, Y, X, C)
            print(f"    Clone image shape: {clone_img.shape}")
            
            # Process clone
            clone_raw, clone_smoothed, clone_info = self.process_clone(
                clone_img,
                clone.bbox,
                gpu_id=gpu_id,
                guide_channel=brain.channel
            )
            
            clone_results.append({
                'name': clone.name,
                'bbox': {
                    'y': clone.bbox.y,
                    'x': clone.bbox.x,
                    'z': clone.bbox.z
                },
                'cells_raw': clone_info['cells_raw'],
                'cells_smoothed': clone_info['cells_smoothed'],
                'time_seconds': clone_info['time_seconds']
            })
            
            clone_masks_raw.append(clone_raw)
            clone_masks_smoothed.append(clone_smoothed)
        
        # Save per-clone masks (full-brain sized, one clone per file)
        print(f"\n  Saving {len(clone_masks_raw)} clone(s) as full-brain masks...")
        brain_shape = (brain_img.shape[0], brain_img.shape[2], brain_img.shape[3])  # (Z, Y, X)
        output_files = []
        
        for i, clone in enumerate(brain.clones):
            bbox = clone.bbox
            z_slice, y_slice, x_slice = bbox.to_slices()
            
            if self.config.output_save_raw:
                full_raw = np.zeros(brain_shape, dtype=np.uint16)
                full_raw[z_slice, y_slice, x_slice] = clone_masks_raw[i]
                raw_path = self.config.get_clone_output_path(brain.name, clone.name, raw=True)
                print(f"    Saving raw: {Path(raw_path).name}")
                save_segmentation_mask(full_raw, raw_path)
                output_files.append(raw_path)
                del full_raw
            
            if self.config.output_save_smoothed:
                full_smoothed = np.zeros(brain_shape, dtype=np.uint16)
                full_smoothed[z_slice, y_slice, x_slice] = clone_masks_smoothed[i]
                smoothed_path = self.config.get_clone_output_path(brain.name, clone.name, raw=False)
                print(f"    Saving smoothed: {Path(smoothed_path).name}")
                save_segmentation_mask(full_smoothed, smoothed_path)
                output_files.append(smoothed_path)
                del full_smoothed
            
            # Add output files to clone results
            clone_results[i]['output_files'] = [f for f in output_files[-(2 if self.config.output_save_raw and self.config.output_save_smoothed else 1):]]
        
        total_cells_raw = sum(cr['cells_raw'] for cr in clone_results)
        total_cells_smoothed = sum(cr['cells_smoothed'] for cr in clone_results)
        print(f"    Total cells: raw={total_cells_raw}, smoothed={total_cells_smoothed}")
        
        elapsed = time.time() - start_time
        print(f"  ✓ {brain.name} complete in {elapsed:.1f}s")
        
        return {
            'status': 'success',
            'clones': clone_results,
            'total_cells_raw': total_cells_raw,
            'total_cells_smoothed': total_cells_smoothed,
            'time_seconds': elapsed,
            'output_files': output_files
        }
    
    def process_clone(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        gpu_id: int = 0,
        guide_channel: int = 0
    ) -> tuple:
        """
        Process a single clone region.
        
        Parameters
        ----------
        image : np.ndarray
            Clone region image (Z, Y, X)
        bbox : BoundingBox
            Bounding box information
        gpu_id : int
            GPU ID to use
            
        Returns
        -------
        tuple
            (raw_segmentation, smoothed_segmentation, info_dict)
        """
        start_time = time.time()
        
        # Run Cellpose on 3 views
        cellpose_results = run_cellpose_3_views(
            image,
            self.cellpose_model,
            channels=self.config.cellpose_channels,
            diameter=self.config.cellpose_diameter,
            flow_threshold=self.config.cellpose_flow_threshold,
            cellprob_threshold=self.config.cellpose_cellprob_threshold
        )
        
        # u-segment3D aggregation
        aggregation_config = {
            'remove_small_objects': self.config.aggregation_remove_small_objects,
            'min_area': self.config.aggregation_min_area,
            'min_prob_thresh': self.config.aggregation_min_prob_thresh,
            'gradient_decay': self.config.aggregation_gradient_decay,
            'n_iter': self.config.aggregation_n_iter
        }
        
        seg_raw = aggregate_direct_method(
            cellpose_results['probs'],
            cellpose_results['flows'],
            aggregation_config
        )
        
        cells_raw = len(np.unique(seg_raw)) - 1
        
        # Apply smoothing if enabled
        if self.config.smoothing_enabled:
            # Extract single-channel guide image for smoothing
            guide_image = image[:, :, :, guide_channel] if image.ndim == 4 else image
            
            smoothing_config = {
                'refine_iters': self.config.smoothing_refine_iters,
                'refine_alpha': self.config.smoothing_refine_alpha,
                'refine_clamp': self.config.smoothing_refine_clamp
            }
            
            seg_smoothed = apply_label_diffusion(
                seg_raw,
                guide_image,
                smoothing_config
            )
        else:
            seg_smoothed = seg_raw
        
        cells_smoothed = len(np.unique(seg_smoothed)) - 1
        elapsed = time.time() - start_time
        
        info = {
            'cells_raw': int(cells_raw),
            'cells_smoothed': int(cells_smoothed),
            'time_seconds': elapsed
        }
        
        return seg_raw, seg_smoothed, info
