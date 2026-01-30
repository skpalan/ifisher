"""ifish_tools: Tools and programs for processing iFISH data."""

__version__ = "0.1.0"

# Lazy imports to avoid dependency issues when running submodules directly
def __getattr__(name):
    if name in ('process_nd2_to_h5', 'process_nd2_folder'):
        from .nd2_processor import process_nd2_to_h5, process_nd2_folder
        return locals()[name]
    elif name in ('run_rsfish_batch', 'find_files_to_process', 
                  'load_brain_timepoint_mapping', 'load_threshold_table', 'RSFishJob'):
        from .rsfish_runner import (
            run_rsfish_batch,
            find_files_to_process,
            load_brain_timepoint_mapping,
            load_threshold_table,
            RSFishJob,
        )
        return locals()[name]
    elif name in ('USeg3DPipeline', 'BoundingBox', 'CloneConfig', 'BrainConfig', 
                  'PipelineConfig', 'generate_config_template'):
        from .usegment3d import (
            USeg3DPipeline,
            BoundingBox,
            CloneConfig,
            BrainConfig,
            PipelineConfig,
            generate_config_template,
        )
        return locals()[name]
    elif name in ('build_count_matrix', 'process_clone'):
        from .countmatrix import build_count_matrix, process_clone
        return locals()[name]
    elif name in ('BoxCoords', 'BrainCloneSpec', 'CloneExtractConfig',
                  'load_bbox_from_mat', 'extract_clone_mask', 'process_brain',
                  'run_pipeline'):
        from .cloneextract import (
            BoxCoords, BrainCloneSpec, CloneExtractConfig,
            load_bbox_from_mat, extract_clone_mask, process_brain, run_pipeline,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
