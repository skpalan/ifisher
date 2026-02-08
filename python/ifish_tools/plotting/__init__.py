"""Plotting module for spatial gene expression 3D visualization.

This module provides tools for creating publication-quality 3D visualizations
of gene expression on unrolled tissue masks.

Example usage:
    from ifish_tools.plotting import process_all_clones

    process_all_clones(
        matrix_dir="/path/to/h5ad/files",
        unroll_dir="/path/to/unroll/output",
        output_dir="/path/to/plots",
        genes=["ase", "Dl", "Hey"],
        workers=4,
    )

CLI usage:
    plot-spatial --matrix-dir ... --unroll-dir ... --output-dir ...
"""

__all__ = [
    "plot_spatial_expression_3d",
    "process_clone_genes",
    "process_all_clones",
    "plot_multi_clone_comparison",
    "process_comparison_all_genes",
    "load_anndata",
    "load_transform_json",
    "load_unrolled_mask",
    "build_label_to_expression_map",
    "build_label_to_annotation_map",
    "get_transformed_centroids",
    "find_matching_clones",
    "find_clones_multi_dir",
]


def __getattr__(name):
    """Lazy import for public API."""
    if name in (
        "plot_spatial_expression_3d",
        "process_clone_genes",
        "process_all_clones",
        "plot_multi_clone_comparison",
        "process_comparison_all_genes",
    ):
        from .core import (
            plot_multi_clone_comparison,
            plot_spatial_expression_3d,
            process_all_clones,
            process_clone_genes,
            process_comparison_all_genes,
        )

        return locals()[name]

    if name in (
        "load_anndata",
        "load_transform_json",
        "load_unrolled_mask",
        "build_label_to_expression_map",
        "build_label_to_annotation_map",
        "get_transformed_centroids",
        "find_matching_clones",
        "find_clones_multi_dir",
    ):
        from .io import (
            build_label_to_annotation_map,
            build_label_to_expression_map,
            find_clones_multi_dir,
            find_matching_clones,
            get_transformed_centroids,
            load_anndata,
            load_transform_json,
            load_unrolled_mask,
        )

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
