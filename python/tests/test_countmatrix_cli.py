"""Tests for countmatrix.cli module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import tifffile

from ifish_tools.countmatrix.cli import parse_args, main


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_parse_args_required_arguments(self):
        """Parse required arguments."""
        argv = [
            "--mask-dir", "/path/to/masks",
            "--puncta-dir", "/path/to/puncta",
            "--output-dir", "/path/to/output",
        ]
        args = parse_args(argv)

        assert args.mask_dir == Path("/path/to/masks")
        assert args.puncta_dir == Path("/path/to/puncta")
        assert args.output_dir == Path("/path/to/output")
        assert args.workers == 1  # default
        assert args.verbose is False  # default
        assert args.mask_pattern == "*_cp_masks.tif"  # default

    def test_parse_args_with_workers(self):
        """Parse with custom worker count."""
        argv = [
            "--mask-dir", "/path/to/masks",
            "--puncta-dir", "/path/to/puncta",
            "--output-dir", "/path/to/output",
            "--workers", "4",
        ]
        args = parse_args(argv)

        assert args.workers == 4

    def test_parse_args_with_verbose(self):
        """Parse with verbose flag."""
        argv = [
            "--mask-dir", "/path/to/masks",
            "--puncta-dir", "/path/to/puncta",
            "--output-dir", "/path/to/output",
            "--verbose",
        ]
        args = parse_args(argv)

        assert args.verbose is True

    def test_parse_args_with_custom_mask_pattern(self):
        """Parse with custom mask pattern."""
        argv = [
            "--mask-dir", "/path/to/masks",
            "--puncta-dir", "/path/to/puncta",
            "--output-dir", "/path/to/output",
            "--mask-pattern", "*.tif",
        ]
        args = parse_args(argv)

        assert args.mask_pattern == "*.tif"

    def test_parse_args_missing_required_fails(self):
        """Fail if required arguments are missing."""
        argv = ["--mask-dir", "/path/to/masks"]
        
        with pytest.raises(SystemExit):
            parse_args(argv)


class TestMain:
    """Test CLI main function."""

    def test_main_validates_mask_directory_exists(self):
        """Exit with error if mask directory does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            output_dir = tmpdir / "output"

            argv = [
                "--mask-dir", str(tmpdir / "nonexistent"),
                "--puncta-dir", str(puncta_dir),
                "--output-dir", str(output_dir),
            ]

            with pytest.raises(SystemExit) as exc_info:
                main(argv)
            
            assert exc_info.value.code == 1

    def test_main_validates_puncta_directory_exists(self):
        """Exit with error if puncta directory does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mask_dir = tmpdir / "masks"
            mask_dir.mkdir()
            output_dir = tmpdir / "output"

            argv = [
                "--mask-dir", str(mask_dir),
                "--puncta-dir", str(tmpdir / "nonexistent"),
                "--output-dir", str(output_dir),
            ]

            with pytest.raises(SystemExit) as exc_info:
                main(argv)
            
            assert exc_info.value.code == 1

    def test_main_validates_mask_files_found(self):
        """Exit with error if no mask files match pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mask_dir = tmpdir / "masks"
            mask_dir.mkdir()
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            output_dir = tmpdir / "output"

            argv = [
                "--mask-dir", str(mask_dir),
                "--puncta-dir", str(puncta_dir),
                "--output-dir", str(output_dir),
            ]

            with pytest.raises(SystemExit) as exc_info:
                main(argv)
            
            assert exc_info.value.code == 1

    def test_main_processes_single_clone_successfully(self):
        """Process single clone successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Setup directories
            mask_dir = tmpdir / "masks"
            mask_dir.mkdir()
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            output_dir = tmpdir / "output"

            # Create mask
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask[0:3, 0:3, 0:3] = 1
            mask_path = mask_dir / "data_brain08_clone1_cp_masks.tif"
            tifffile.imwrite(mask_path, mask)

            # Create puncta CSV
            puncta_df = pd.DataFrame(
                {
                    "x": [1.0, 1.5],
                    "y": [1.0, 1.5],
                    "z": [1.0, 1.5],
                    "t": [0, 0],
                    "c": [0, 0],
                    "intensity": [100, 100],
                }
            )
            csv_path = puncta_dir / "data_brain08_gene-test.csv"
            puncta_df.to_csv(csv_path, index=False)

            argv = [
                "--mask-dir", str(mask_dir),
                "--puncta-dir", str(puncta_dir),
                "--output-dir", str(output_dir),
            ]

            # Should not raise
            main(argv)

            # Verify output file was created
            expected_output = output_dir / "brain08_clone1.h5ad"
            assert expected_output.exists()

    def test_main_processes_multiple_clones_sequentially(self):
        """Process multiple clones with workers=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Setup directories
            mask_dir = tmpdir / "masks"
            mask_dir.mkdir()
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            output_dir = tmpdir / "output"

            # Create two masks
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask[0:3, 0:3, 0:3] = 1
            
            mask1_path = mask_dir / "data_brain08_clone1_cp_masks.tif"
            tifffile.imwrite(mask1_path, mask)
            
            mask2_path = mask_dir / "data_brain08_clone2_cp_masks.tif"
            tifffile.imwrite(mask2_path, mask)

            # Create puncta CSVs
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
            csv_path = puncta_dir / "data_brain08_gene-test.csv"
            puncta_df.to_csv(csv_path, index=False)

            argv = [
                "--mask-dir", str(mask_dir),
                "--puncta-dir", str(puncta_dir),
                "--output-dir", str(output_dir),
            ]

            main(argv)

            # Verify both output files were created
            assert (output_dir / "brain08_clone1.h5ad").exists()
            assert (output_dir / "brain08_clone2.h5ad").exists()

    def test_main_handles_processing_errors_gracefully(self):
        """Continue processing other clones if one fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Setup directories
            mask_dir = tmpdir / "masks"
            mask_dir.mkdir()
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            output_dir = tmpdir / "output"

            # Create one good mask and one bad mask (empty)
            good_mask = np.zeros((10, 10, 10), dtype=np.uint16)
            good_mask[0:3, 0:3, 0:3] = 1
            good_mask_path = mask_dir / "data_brain08_clone1_cp_masks.tif"
            tifffile.imwrite(good_mask_path, good_mask)

            bad_mask = np.zeros((10, 10, 10), dtype=np.uint16)
            bad_mask_path = mask_dir / "data_brain09_clone1_cp_masks.tif"
            tifffile.imwrite(bad_mask_path, bad_mask)

            # Create puncta for brain08 only
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
            csv_path = puncta_dir / "data_brain08_gene-test.csv"
            puncta_df.to_csv(csv_path, index=False)

            argv = [
                "--mask-dir", str(mask_dir),
                "--puncta-dir", str(puncta_dir),
                "--output-dir", str(output_dir),
            ]

            # Should not crash
            main(argv)

            # Good one should still be processed
            assert (output_dir / "brain08_clone1.h5ad").exists()
            # Bad one should not create output
            assert not (output_dir / "brain09_clone1.h5ad").exists()

    def test_main_uses_parallel_workers(self):
        """Process clones in parallel with multiple workers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Setup directories
            mask_dir = tmpdir / "masks"
            mask_dir.mkdir()
            puncta_dir = tmpdir / "puncta"
            puncta_dir.mkdir()
            output_dir = tmpdir / "output"

            # Create two masks
            mask = np.zeros((10, 10, 10), dtype=np.uint16)
            mask[0:3, 0:3, 0:3] = 1
            
            mask1_path = mask_dir / "data_brain08_clone1_cp_masks.tif"
            tifffile.imwrite(mask1_path, mask)
            
            mask2_path = mask_dir / "data_brain08_clone2_cp_masks.tif"
            tifffile.imwrite(mask2_path, mask)

            # Create puncta CSVs
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
            csv_path = puncta_dir / "data_brain08_gene-test.csv"
            puncta_df.to_csv(csv_path, index=False)

            argv = [
                "--mask-dir", str(mask_dir),
                "--puncta-dir", str(puncta_dir),
                "--output-dir", str(output_dir),
                "--workers", "2",
            ]

            main(argv)

            # Verify both output files were created
            assert (output_dir / "brain08_clone1.h5ad").exists()
            assert (output_dir / "brain08_clone2.h5ad").exists()
