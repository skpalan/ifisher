"""Tests for countmatrix.core module."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
import tifffile

from ifish_tools.countmatrix.core import (
    extract_gene_from_filename,
    extract_brain_from_mask,
    extract_clone_from_mask,
    compute_cell_metadata,
    assign_puncta_to_cells,
    build_count_matrix,
    process_clone,
)


class TestFilenameExtraction:
    """Test filename parsing functions."""

    def test_extract_gene_from_filename(self):
        """Extract gene name from puncta CSV filename."""
        filename = "Gel20251024_brain08_round01_gene-dati.csv"
        assert extract_gene_from_filename(filename) == "dati"

    def test_extract_gene_with_complex_name(self):
        """Extract gene name with hyphens (stops at underscore/dot)."""
        filename = "data_gene-foo-bar_baz.csv"
        assert extract_gene_from_filename(filename) == "foo-bar"

    def test_extract_gene_raises_on_invalid_filename(self):
        """Raise ValueError if gene pattern not found."""
        with pytest.raises(ValueError, match="Cannot extract gene name"):
            extract_gene_from_filename("no_gene_pattern.csv")

    def test_extract_brain_from_mask(self):
        """Extract brain ID from mask filename."""
        filename = "Gel20251024_round00_brain08_intact_cropped_clone1_useg_0129_cp_masks.tif"
        assert extract_brain_from_mask(filename) == "08"

    def test_extract_brain_with_different_id(self):
        """Extract different brain ID."""
        filename = "data_brain12_other.tif"
        assert extract_brain_from_mask(filename) == "12"

    def test_extract_brain_raises_on_invalid_filename(self):
        """Raise ValueError if brain pattern not found."""
        with pytest.raises(ValueError, match="Cannot extract brain ID"):
            extract_brain_from_mask("no_brain_pattern.tif")

    def test_extract_clone_from_mask(self):
        """Extract clone ID from mask filename."""
        filename = "Gel20251024_round00_brain08_intact_cropped_clone1_useg_0129_cp_masks.tif"
        assert extract_clone_from_mask(filename) == "1"

    def test_extract_clone_with_different_id(self):
        """Extract different clone ID."""
        filename = "data_clone5_other.tif"
        assert extract_clone_from_mask(filename) == "5"

    def test_extract_clone_raises_on_invalid_filename(self):
        """Raise ValueError if clone pattern not found."""
        with pytest.raises(ValueError, match="Cannot extract clone ID"):
            extract_clone_from_mask("no_clone_pattern.tif")


class TestComputeCellMetadata:
    """Test cell metadata computation from masks."""

    def test_compute_metadata_single_cell(self):
        """Compute metadata for single cell."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        mask[2:5, 3:7, 4:8] = 1  # Single cell

        meta = compute_cell_metadata(mask)

        assert len(meta) == 1
        assert meta["label"].iloc[0] == 1
        assert meta["cell_size"].iloc[0] == 3 * 4 * 4  # 48 voxels
        # Check centroids are reasonable
        assert 2 <= meta["centroid_z"].iloc[0] <= 5
        assert 3 <= meta["centroid_y"].iloc[0] <= 7
        assert 4 <= meta["centroid_x"].iloc[0] <= 8

    def test_compute_metadata_multiple_cells(self):
        """Compute metadata for multiple cells."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        mask[0:3, 0:3, 0:3] = 1  # Cell 1
        mask[5:8, 5:8, 5:8] = 2  # Cell 2

        meta = compute_cell_metadata(mask)

        assert len(meta) == 2
        assert list(meta["label"].values) == [1, 2]
        assert meta["cell_size"].iloc[0] == 27
        assert meta["cell_size"].iloc[1] == 27

    def test_compute_metadata_empty_mask(self):
        """Return empty DataFrame for mask with no cells."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)

        meta = compute_cell_metadata(mask)

        assert len(meta) == 0
        assert list(meta.columns) == [
            "label",
            "centroid_z",
            "centroid_y",
            "centroid_x",
            "cell_size",
        ]

    def test_compute_metadata_skips_background(self):
        """Background (label 0) should be excluded."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        mask[0:5, 0:5, 0:5] = 1  # Only cell 1

        meta = compute_cell_metadata(mask)

        assert len(meta) == 1
        assert 0 not in meta["label"].values


class TestAssignPunctaToCells:
    """Test puncta assignment to cells."""

    def test_assign_single_punctum_to_cell(self):
        """Assign punctum inside cell."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        mask[2:5, 3:7, 4:8] = 1  # Cell 1

        puncta_df = pd.DataFrame({"x": [5.0], "y": [4.0], "z": [3.0]})

        assignments = assign_puncta_to_cells(puncta_df, mask)

        assert len(assignments) == 1
        assert assignments[0] == 1

    def test_assign_multiple_puncta(self):
        """Assign multiple puncta to different cells."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        mask[0:3, 0:3, 0:3] = 1  # Cell 1
        mask[5:8, 5:8, 5:8] = 2  # Cell 2

        puncta_df = pd.DataFrame(
            {
                "x": [1.0, 6.0, 9.0],
                "y": [1.0, 6.0, 9.0],
                "z": [1.0, 6.0, 9.0],
            }
        )

        assignments = assign_puncta_to_cells(puncta_df, mask)

        assert len(assignments) == 3
        assert assignments[0] == 1  # In cell 1
        assert assignments[1] == 2  # In cell 2
        assert assignments[2] == 0  # Outside cells (background)

    def test_assign_clips_out_of_bounds(self):
        """Clips coordinates outside mask bounds."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        mask[0:3, 0:3, 0:3] = 1

        puncta_df = pd.DataFrame(
            {
                "x": [100.0, 1.0],  # First out of bounds
                "y": [100.0, 1.0],
                "z": [100.0, 1.0],
            }
        )

        assignments = assign_puncta_to_cells(puncta_df, mask)

        # Should not crash, should clip to bounds
        assert len(assignments) == 2
        assert assignments[1] == 1  # Second punctum is in cell


class TestBuildCountMatrix:
    """Test count matrix building."""

    def test_build_count_matrix_single_gene(self):
        """Build count matrix with single gene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mask
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask[0:3, 0:3, 0:3] = 1  # Cell 1
            mask[5:8, 5:8, 5:8] = 2  # Cell 2
            mask_path = tmpdir / "Gel20251024_round00_brain08_intact_cropped_clone1_useg_0129_cp_masks.tif"
            tifffile.imwrite(mask_path, mask)

            # Create puncta CSV
            puncta_df = pd.DataFrame(
                {
                    "x": [1.0, 1.5, 6.0],
                    "y": [1.0, 1.5, 6.0],
                    "z": [1.0, 1.5, 6.0],
                    "t": [0, 0, 0],
                    "c": [0, 0, 0],
                    "intensity": [100, 100, 100],
                }
            )
            csv_path = tmpdir / "Gel20251024_brain08_round01_gene-dati.csv"
            puncta_df.to_csv(csv_path, index=False)

            # Build count matrix
            adata = build_count_matrix(mask_path, [csv_path])

            assert adata is not None
            assert adata.n_obs == 2  # 2 cells
            assert adata.n_vars == 1  # 1 gene
            assert list(adata.var_names) == ["dati"]
            assert adata.X[0, 0] == 2  # Cell 1 has 2 puncta
            assert adata.X[1, 0] == 1  # Cell 2 has 1 punctum

            # Check obs
            assert "brain_id" in adata.obs.columns
            assert "clone_id" in adata.obs.columns
            assert all(adata.obs["brain_id"] == "08")
            assert all(adata.obs["clone_id"] == "1")

    def test_build_count_matrix_multiple_genes(self):
        """Build count matrix with multiple genes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mask
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask[0:3, 0:3, 0:3] = 1
            mask_path = tmpdir / "data_brain08_clone1_masks.tif"
            tifffile.imwrite(mask_path, mask)

            # Create puncta CSVs for two genes
            # Include some puncta at high z to ensure z_scale=1
            puncta1 = pd.DataFrame(
                {
                    "x": [1.0, 1.5, 5.0],  # Last one outside cell for z_scale detection
                    "y": [1.0, 1.5, 5.0],
                    "z": [1.0, 1.5, 9.0],  # max_z=9 ensures z_scale=1
                    "t": [0, 0, 0],
                    "c": [0, 0, 0],
                    "intensity": [100, 100, 100],
                }
            )
            csv1 = tmpdir / "data_gene-geneA.csv"
            puncta1.to_csv(csv1, index=False)

            puncta2 = pd.DataFrame(
                {
                    "x": [1.0, 1.5, 2.0, 5.0],  # Last one outside cell for z_scale detection
                    "y": [1.0, 1.5, 2.0, 5.0],
                    "z": [1.0, 1.5, 2.0, 9.0],  # max_z=9 ensures z_scale=1
                    "t": [0, 0, 0, 0],
                    "c": [0, 0, 0, 0],
                    "intensity": [100, 100, 100, 100],
                }
            )
            csv2 = tmpdir / "data_gene-geneB.csv"
            puncta2.to_csv(csv2, index=False)

            # Build count matrix
            adata = build_count_matrix(mask_path, [csv1, csv2])

            assert adata is not None
            assert adata.n_obs == 1
            assert adata.n_vars == 2
            assert list(adata.var_names) == ["geneA", "geneB"]
            assert adata.X[0, 0] == 2  # geneA: 2 puncta
            assert adata.X[0, 1] == 3  # geneB: 3 puncta

    def test_build_count_matrix_empty_mask(self):
        """Return None for empty mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create empty mask
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask_path = tmpdir / "data_brain08_clone1_masks.tif"
            tifffile.imwrite(mask_path, mask)

            # Create dummy puncta
            puncta_df = pd.DataFrame(
                {
                    "x": [1.0],
                    "y": [1.0],
                    "z": [1.0],
                    "t": [0],
                    "c": [0],
                    "intensity": [100],
                }
            )
            csv_path = tmpdir / "data_gene-test.csv"
            puncta_df.to_csv(csv_path, index=False)

            # Should return None
            adata = build_count_matrix(mask_path, [csv_path])

            assert adata is None


class TestProcessClone:
    """Test process_clone function."""

    def test_process_clone_creates_h5ad(self):
        """Process clone and save h5ad file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mask
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask[0:3, 0:3, 0:3] = 1
            mask[5:8, 5:8, 5:8] = 2
            mask_path = tmpdir / "Gel20251024_round00_brain08_intact_cropped_clone1_useg_0129_cp_masks.tif"
            tifffile.imwrite(mask_path, mask)

            # Create puncta CSV
            puncta_df = pd.DataFrame(
                {
                    "x": [1.0, 6.0],
                    "y": [1.0, 6.0],
                    "z": [1.0, 6.0],
                    "t": [0, 0],
                    "c": [0, 0],
                    "intensity": [100, 100],
                }
            )
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            csv_path = puncta_dir / "Gel20251024_brain08_round01_gene-test.csv"
            puncta_df.to_csv(csv_path, index=False)

            output_dir = tmpdir / "output"

            # Process clone
            result_path = process_clone(mask_path, puncta_dir, output_dir)

            assert result_path is not None
            assert result_path.exists()
            assert result_path.name == "brain08_clone1.h5ad"

            # Verify h5ad can be loaded
            import anndata as ad

            adata = ad.read_h5ad(result_path)
            assert adata.n_obs == 2
            assert adata.n_vars == 1

    def test_process_clone_no_puncta_returns_none(self):
        """Return None if no matching puncta CSVs found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mask
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask[0:3, 0:3, 0:3] = 1
            mask_path = tmpdir / "data_brain08_clone1_masks.tif"
            tifffile.imwrite(mask_path, mask)

            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            output_dir = tmpdir / "output"

            # Process clone with no puncta
            result_path = process_clone(mask_path, puncta_dir, output_dir)

            assert result_path is None

    def test_process_clone_empty_mask_returns_none(self):
        """Return None if mask has no cells."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create empty mask
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask_path = tmpdir / "data_brain08_clone1_masks.tif"
            tifffile.imwrite(mask_path, mask)

            # Create puncta CSV
            puncta_df = pd.DataFrame(
                {
                    "x": [1.0],
                    "y": [1.0],
                    "z": [1.0],
                    "t": [0],
                    "c": [0],
                    "intensity": [100],
                }
            )
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            csv_path = puncta_dir / "data_brain08_gene-test.csv"
            puncta_df.to_csv(csv_path, index=False)

            output_dir = tmpdir / "output"

            # Process clone with empty mask
            result_path = process_clone(mask_path, puncta_dir, output_dir)

            assert result_path is None
