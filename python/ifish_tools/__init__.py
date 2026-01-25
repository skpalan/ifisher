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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
