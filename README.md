# ifisher

**ifisher** is a comprehensive toolkit for processing iFISH (immunofluorescence in situ hybridization) microscopy data. It provides end-to-end data processing from ND2 file conversion to 3D cell segmentation and clone analysis.

## Features

- **ND2 Processing**: High-performance parallel conversion of ND2 files to HDF5/TIFF formats
- **RS-FISH Integration**: Batch processing wrapper for [RS-FISH](https://github.com/PreibischLab/RS-FISH) spot detection
- **3D Segmentation**: State-of-the-art 3D cell segmentation using [u-Segment3D](https://github.com/DanuserLab/u-segment3D) with Cellpose
- **Clone Unrolling**: 3D clone transformation via principal curve analysis
- **Spatial Visualization**: Publication-ready 3D expression heatmaps with customizable colormaps

## Installation

### As an Lmod Module (Recommended for HPC)

If you're on a system with Lmod module support:

```bash
ml ifisher/0.1.0
```

This automatically loads all dependencies including the RS-FISH module.

### From Source

```bash
# Clone the repository
git clone https://github.com/skpalan/ifisher.git
cd ifisher/python

# Create a virtual environment (Python 3.11+ recommended)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Available Tools

### 1. ND2 Processor

Convert ND2 microscopy files to HDF5 or TIFF format with parallel processing.

**Command Line:**
```bash
nd2-processor input.nd2 output_dir/ \
  --h5-16bit round00 round01 \
  --workers 20 \
  --compression gzip-4
```

**Python API:**
```python
from ifish_tools.nd2_processor import process_nd2_to_h5, process_nd2_folder

# Single file
process_nd2_to_h5(
    nd2_path="data/brain01.nd2",
    output_dir="output/",
    h5_16bit=["round00", "round01"],
    compression="gzip-4",
)

# Batch folder
process_nd2_folder(
    nd2_folder="raw_data/",
    output_dir="output/",
    h5_16bit=["round00", "round01"],
    workers=20,
)
```

**Key Features:**
- Parallel processing with configurable file and channel workers
- Multiple output formats (HDF5 16/8-bit, TIFF 16/8-bit)
- Automatic channel detection
- Configurable memory limit
- Progress tracking with tqdm

### 2. RS-FISH Runner

Batch processing wrapper for RS-FISH spot detection with automatic parameter management.

**Command Line:**
```bash
rsfish-runner \
  --data-dir /path/to/h5_files \
  --metadata nd2_metadata.csv \
  --thresholds threshold_table.csv \
  --output-dir /path/to/output \
  --sigma 1.3 \
  --workers 20
```

**Python API:**
```python
from ifish_tools.rsfish_runner import (
    load_brain_timepoint_mapping,
    load_threshold_table,
    find_files_to_process,
    run_rsfish_batch,
)

# Build job list
brain_tp = load_brain_timepoint_mapping("nd2_metadata.csv")
thresholds = load_threshold_table("threshold_table.csv")
jobs = find_files_to_process(
    data_dir="/path/to/h5_files",
    brain_timepoints=brain_tp,
    thresholds=thresholds,
    output_dir="/path/to/output",
)

# Run batch
results = run_rsfish_batch(
    jobs=jobs,
    output_dir="/path/to/output",
    max_workers=20,
    sigma=1.3,
)
```

**Key Features:**
- Automatic brain-to-timepoint mapping
- Threshold lookup per round/channel/timepoint
- Parallel execution
- Dry-run mode for testing
- Progress tracking

### 3. u-Segment3D Pipeline

3D cell segmentation by aggregating 2D segmentations from orthogonal views.

**Command Line:**
```bash
# Generate configuration template
usegment3d --generate-config config.yaml

# Run pipeline
usegment3d --config config.yaml --gpus 0,1

# Dry run to validate
usegment3d --config config.yaml --dry-run
```

**Python API:**
```python
from ifish_tools.usegment3d import USeg3DPipeline

# Load from YAML config
pipeline = USeg3DPipeline.from_yaml('config.yaml')

# Run segmentation
results = pipeline.run(
    gpus=[0, 1],
    skip_existing=True
)
```

**Key Features:**
- Integrates Cellpose 2D segmentation
- u-Segment3D consensus 3D aggregation (direct & indirect methods)
- Label diffusion smoothing
- GPU acceleration
- Configurable via YAML
- Batch processing of multiple brains

### 4. Clone Extract

Extract per-clone segmentation masks from whole-brain volumes with morphological closing to fill gaps between cells.

**Command Line:**
```bash
# Validate configuration
clone-extract --config config.yaml --dry-run

# Run with GPU acceleration (recommended)
ml ifisher cuda/12.9.1
clone-extract --config config.yaml --gpu

# Override closing radius
clone-extract --config config.yaml --gpu --closing-radius 8

# CPU with parallel workers
clone-extract --config config.yaml --workers 10
```

**YAML Configuration:**
```yaml
output_dir: "/path/to/output"
closing_radius: 5
date_tag: "0202"

brains:
  - name: "brain08"
    mask_path: "/path/to/brain08_cp_masks.tif"
    bbox: "/path/to/brain08/ref/bbox_ref.mat"
    clones:
      clone1: [560, 788, 775, 1210, 71, 420]  # [ymin, ymax, xmin, xmax, zmin, zmax]
```

**Python API:**
```python
from ifish_tools.cloneextract import CloneExtractConfig, run_pipeline

config = CloneExtractConfig.from_yaml("config.yaml")
output_paths = run_pipeline(config, use_gpu=True)
```

**Key Features:**
- GPU-accelerated morphological closing via CuPy
- Per-label closing with distance-transform overlap resolution
- Automatic 2x Z resolution detection and scaling
- Multi-worker brain-level parallelism + threaded label-level closing
- YAML-driven batch processing
- zstd-compressed TIFF output

### 5. Clone Unrolling

Unroll 3D tissue clones along their principal curve using [ElPiGraph](https://github.com/j-bac/elpigraph-python) elastic principal curves and TopoVelo-style spherical coordinate transformation. Automatically detects the neuroblast (largest cell) as the starting endpoint.

**Command Line:**
```bash
# Basic usage
unroll \
  --mask-dir masks/ \
  --puncta-dir puncta/pixel/ \
  --output-dir output/ \
  --n-anchors 30 \
  --plane zx

# GPU-accelerated with parallel workers (recommended)
ml cuda/12.9.1 ifisher
unroll \
  --mask-dir masks/ \
  --output-dir output/ \
  --gpu --workers 4
```

**Key Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mask-dir` | (required) | Directory with 3D mask TIFFs |
| `--puncta-dir` | (optional) | Directory with puncta CSVs (x,y,z columns) |
| `--output-dir` | (required) | Output directory |
| `--n-anchors` | 30 | Number of anchor points on the principal curve |
| `--plane` | zx | Unrolling plane (`xy`, `yz`, or `zx`) |
| `--padding` | 50 | Padding around unrolled mask in voxels |
| `--epg-mu` | 1.0 | ElPiGraph stretching penalty |
| `--epg-lambda` | 0.01 | ElPiGraph bending penalty |
| `--mask-pattern` | *.tif | Glob pattern for mask files |
| `--gpu` | auto | Enable GPU acceleration (auto-detects CuPy) |
| `--no-gpu` | - | Force CPU-only processing |
| `--device` | 0 | GPU device ID |
| `--workers`, `-j` | 1 | Number of parallel workers for multi-mask processing |

**GPU Acceleration:**
- Requires CUDA module: `ml cuda/12.9.1`
- Uses CuPy for vectorized mask transformation (~10-50x speedup for transform step)
- Auto-detects GPU availability; falls back to CPU if unavailable

**Parallel Processing:**
- Use `--workers N` to process multiple masks concurrently
- I/O-bound workload benefits from threading (overlaps file loading with computation)
- ~2-3x speedup with 3-4 workers

**Output structure:**
```
output_dir/brain{ID}_clone{N}/
  unrolled_mask.tif      # Transformed 3D mask
  transform.json         # Transformation parameters
  qc_plot.png            # QC plot (centroids + mask voxels, before/after)
  puncta/*.csv           # Transformed puncta per round
```

### 6. Spatial Expression Plotting

Generate publication-ready 3D spatial expression heatmaps for unrolled tissues.

**Command Line:**
```bash
python -m ifish_tools.plotting.cli \
  --matrix-dir /path/to/h5ad \
  --unroll-dir /path/to/unrolled \
  --raw-mask-dir /path/to/raw_masks \
  --output-dir /path/to/plots \
  --workers 12 \
  --cmap viridis
```

**Key Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--matrix-dir` | (required) | Directory with .h5ad count matrix files |
| `--unroll-dir` | (required) | Directory with unrolled outputs |
| `--raw-mask-dir` | auto | Directory with raw (non-unrolled) masks |
| `--output-dir` | (required) | Output directory for plots |
| `--genes` | all | Specific genes to plot |
| `--cmap` | viridis | Matplotlib colormap for expression |
| `--dpi` | 300 | Output resolution |
| `--workers`, `-j` | 1 | Parallel workers |

**Output:**
- 2×3 panel PNG per gene per clone:
  - Row 1: Unrolled tissue (random colors, expression heatmap, centroids)
  - Row 2: Raw tissue (same 3 panels)

**Python API:**
```python
from ifish_tools.plotting import (
    plot_spatial_expression_3d,
    process_all_clones,
)

# Generate all plots
outputs = process_all_clones(
    matrix_dir="matrices/",
    unroll_dir="unrolled/",
    raw_mask_dir="masks/",
    output_dir="plots/",
    cmap="viridis",
    workers=12,
)
```

### 7. Multi-Clone Comparison Plotting

Generate side-by-side comparison figures with multiple clones per gene. Supports expression heatmaps, pseudocolor, and cell-type annotation panels.

**Python API:**
```python
from ifish_tools.plotting import process_comparison_all_genes

# Generate comparison plots for selected genes across clones
results = process_comparison_all_genes(
    matrix_dirs=["/path/to/h5ad_dir1", "/path/to/h5ad_dir2"],
    unroll_dirs=["/path/to/unroll_dir1", "/path/to/unroll_dir2"],
    output_dir="/path/to/output",
    clones=["brain08_clone1", "brain09_clone1", "brain10_clone1"],
    genes=["Imp", "dati", "pdm3"],         # None = all genes
    panels=[
        ("expression", "centroids"),        # expression centroid scatter
        ("annotation", "centroids"),        # cell-type annotation scatter
    ],
    annotation_col="naive_cell_type",       # obs column for annotations
    color_scale="per_clone",                # "shared" or "per_clone"
    panel_height=4,
    fontsize=9,
    cmap="viridis",
    dpi=300,
    workers=12,
)
```

**Low-Level API (single gene):**
```python
from ifish_tools.plotting import plot_multi_clone_comparison

# Pre-loaded clone data dicts
plot_multi_clone_comparison(
    clone_data=[clone1_dict, clone2_dict],
    gene_name="Imp",
    output_path="Imp_comparison.png",
    panels=[("expression", "centroids"), ("annotation", "centroids")],
    color_scale="per_clone",
    annotation_col="naive_cell_type",
)
```

**Panel Types:**

| Panel | Render Mode | Description |
|-------|-------------|-------------|
| `expression` | `voxels` | Expression-colored voxel heatmap |
| `expression` | `centroids` | Expression-colored centroid scatter |
| `pseudocolor` | `voxels` | Random-color per-cell voxels |
| `pseudocolor` | `centroids` | Random-color centroid scatter |
| `annotation` | `voxels` | Cell-type colored voxels |
| `annotation` | `centroids` | Cell-type colored centroid scatter |

**Key Features:**
- Flexible multi-directory clone discovery (annotated + plain h5ad files)
- Shared or per-clone color normalization with NB outlier exclusion
- Configurable panel layout (rows = panel types, columns = clones)
- Automatic aspect ratio inference from mask bounding box
- Graceful degradation for unannotated clones
- Parallel gene processing via ProcessPoolExecutor

**Python API (Unroll):**
```python
from ifish_tools.unroll import (
    compute_centroids,
    fit_principal_curve,
    sort_anchors,
    unroll_clone,
    transform_mask,
    detect_endpoints,
    find_endpoint_anchors,
)
from ifish_tools.unroll.io import load_mask

# Load mask and compute centroids
mask = load_mask("mask.tif")
centroids = compute_centroids(mask)

# Detect endpoints (neuroblast = largest cell)
start_cell, end_cell = detect_endpoints(mask, centroids)

# Fit elastic principal curve
anchor_pos, edges, node_degree, assignments = fit_principal_curve(
    centroids, n_anchors=30
)

# Sort anchors from start to end
start_anchor, end_anchor = find_endpoint_anchors(
    anchor_pos, node_degree, centroids[start_cell], centroids[end_cell]
)
anchors_ordered = sort_anchors(edges, start_anchor, end_anchor)

# Unroll centroids (spherical) and get rigid transforms for masks
new_centroids, transform_params = unroll_clone(
    centroids, anchors_ordered, anchor_pos, assignments, start_cell, plane="zx"
)

# Transform mask (rigid rotation per cell)
# CPU version
unrolled_mask, offset = transform_mask(mask, transform_params, padding=50)

# GPU-accelerated version
unrolled_mask, offset = transform_mask(mask, transform_params, padding=50, use_gpu=True)
```

## Module Structure

```
ifish_tools/
├── __init__.py              # Package initialization
├── nd2_processor.py         # ND2 → HDF5/TIFF conversion
├── rsfish_runner.py         # RS-FISH batch processing
├── usegment3d/             # 3D segmentation pipeline
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration classes
│   ├── pipeline.py         # Main pipeline orchestration
│   ├── cellpose_views.py   # Cellpose on orthogonal views
│   ├── aggregation.py      # u-Segment3D aggregation
│   ├── smoothing.py        # Label diffusion smoothing
│   └── utils.py            # Utility functions
├── cloneextract/          # Clone mask extraction
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration & YAML parsing
│   └── core.py            # Extraction & morphological closing (CPU/GPU)
├── countmatrix/           # Gene-by-cell count matrices
│   ├── __init__.py
│   ├── cli.py             # count-matrix CLI
│   ├── qc_cli.py          # count-qc CLI
│   ├── core.py            # Matrix assembly
│   ├── regis_puncta.py    # Registered puncta lookup
│   └── qc.py              # Quality-control metrics
├── unroll/                # Clone unrolling
│   ├── __init__.py
│   ├── __main__.py         # python -m entry point
│   ├── cli.py              # Command-line interface + QC plots
│   ├── endpoints.py        # Auto endpoint detection
│   ├── principal_curve.py  # ElPiGraph principal curve fitting
│   ├── transform.py        # Spherical unrolling + rigid rotation
│   └── io.py               # I/O utilities
└── plotting/              # Spatial expression visualization
    ├── __init__.py
    ├── cli.py              # Command-line interface
    ├── core.py             # 3D plotting functions
    └── io.py               # Data loading utilities
```

## Dependencies

### Core
- Python ≥ 3.11
- numpy, scipy, pandas
- h5py, tifffile, nd2
- pyyaml, tqdm, psutil

### Deep Learning
- torch ≥ 2.0
- cellpose ≥ 3.1
- u-Segment3D 0.1.4

### Curve Fitting
- elpigraph-python

### GPU Acceleration (Optional)
- cupy-cuda12x (for GPU-accelerated unrolling)

### Image Processing
- scikit-image
- opencv-python-headless
- matplotlib

### External Tools
- [RS-FISH](https://github.com/PreibischLab/RS-FISH) (for spot detection)

## Environment Setup

The ifisher module is available on HPC systems with Lmod:

```bash
# Load the module
ml ifisher/0.1.0

# This automatically:
# - Loads rs-fish module dependency
# - Activates the ifisher Python environment
# - Adds CLI tools to PATH
# - Makes Python packages importable
```

**Environment Variables Set:**
- `IFISHER_HOME`: Path to ifisher installation
- `PATH`: Includes ifisher/bin for CLI tools

## Usage Examples

### Complete Pipeline Example

```bash
# 1. Load module
ml ifisher

# 2. Convert ND2 to HDF5
nd2-processor brain01.nd2 output/ \
  --h5-16bit round00 round01 round02 \
  --workers 20

# 3. Run RS-FISH spot detection
rsfish-runner \
  --data-dir output/h5_16bit \
  --metadata metadata.csv \
  --thresholds thresholds.csv \
  --output-dir output/puncta \
  --workers 20

# 4. Run 3D segmentation
usegment3d --generate-config segmentation_config.yaml
# Edit config file as needed
usegment3d --config segmentation_config.yaml --gpus 0,1

# 5. Unroll clones along principal curves
unroll --mask-dir masks/ --puncta-dir output/puncta/ --output-dir output/unrolled/
```

### Python Scripting Example

```python
import numpy as np
from ifish_tools.nd2_processor import process_nd2_folder
from ifish_tools.usegment3d import USeg3DPipeline
from ifish_tools.unroll import fit_principal_curve, unroll_clone

# 1. Batch process ND2 files
process_nd2_folder(
    input_folder="raw_data/",
    output_folder="processed/",
    rounds_16bit=["round00", "round01"],
    workers=20
)

# 2. Run 3D segmentation
pipeline = USeg3DPipeline.from_yaml('config.yaml')
results = pipeline.run(gpus=[0])

# 3. Unroll clones
# See "Clone Unrolling" section above for Python API usage
```

## Citation

If you use ifisher in your research, please cite:

**u-Segment3D:**
```
Felix Y. Zhou, et al. (2025)
"Universal consensus 3D segmentation of cells from 2D segmented stacks"
Nature Methods
doi:10.1038/s41592-025-02887-w
```

**RS-FISH:**
```
Ella Bahry, Laura Breimann, et al. (2022)
"RS-FISH: Precise, interactive, fast, and scalable FISH spot detection"
Nature Methods
doi:10.1038/s41592-022-01669-y
```

**Cellpose:**
```
Carsen Stringer, et al. (2021)
"Cellpose: a generalist algorithm for cellular segmentation"
Nature Methods
doi:10.1038/s41592-020-01018-x
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See [LICENSE](LICENSE) file for details.

## Authors

- iFISH Lab
- Repository: https://github.com/skpalan/ifisher

## Support

For questions or issues:
- Open a GitHub issue
- Contact the maintainers

---

**Version:** 0.1.0
**Last Updated:** February 2026
