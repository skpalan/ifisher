# ifisher

**ifisher** is a comprehensive toolkit for processing iFISH (immunofluorescence in situ hybridization) microscopy data. It provides end-to-end data processing from ND2 file conversion to 3D cell segmentation and clone analysis.

## Features

- **ND2 Processing**: High-performance parallel conversion of ND2 files to HDF5/TIFF formats
- **RS-FISH Integration**: Batch processing wrapper for [RS-FISH](https://github.com/PreibischLab/RS-FISH) spot detection
- **3D Segmentation**: State-of-the-art 3D cell segmentation using [u-Segment3D](https://github.com/DanuserLab/u-segment3D) with Cellpose
- **Clone Unrolling**: 3D clone transformation via principal curve analysis

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
from ifish_tools.nd2_processor import process_nd2_to_h5

process_nd2_to_h5(
    nd2_path="data/brain01.nd2",
    output_dir="output/",
    rounds_16bit=["round00", "round01"],
    workers=20
)
```

**Key Features:**
- Parallel processing with configurable workers
- Multiple output formats (HDF5 16/8-bit, TIFF 16/8-bit)
- Automatic channel detection
- Memory-efficient processing
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
from ifish_tools.rsfish_runner import run_rsfish_batch

results = run_rsfish_batch(
    data_dir="/path/to/h5_files",
    metadata_csv="nd2_metadata.csv",
    threshold_csv="threshold_table.csv",
    output_dir="/path/to/output",
    sigma=1.3,
    workers=20
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

### 4. Clone Unrolling

Transform 3D clones along principal curves for spatial analysis.

**Python API:**
```python
from ifish_tools.unroll import (
    compute_centroids,
    fit_principal_curve,
    unroll_clone,
    transform_mask
)

# Compute principal curve through cell centroids
centroids = compute_centroids(mask_3d)
anchors, tangents = fit_principal_curve(centroids, n_anchors=50)

# Unroll the clone
unrolled_coords = unroll_clone(
    coords=cell_coords,
    anchors=anchors,
    tangents=tangents
)

# Transform the mask
unrolled_mask = transform_mask(mask_3d, anchors, tangents)
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
└── unroll/                 # Clone unrolling
    ├── __init__.py
    ├── principal_curve.py  # Principal curve fitting
    ├── transform.py        # Coordinate transformations
    └── io.py              # I/O utilities
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

# 3. Unroll clones for analysis
for brain, result in results.items():
    mask = result['segmentation']
    centroids = compute_centroids(mask)
    anchors, tangents = fit_principal_curve(centroids)
    unrolled = unroll_clone(centroids, anchors, tangents)
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
**Last Updated:** January 2026
