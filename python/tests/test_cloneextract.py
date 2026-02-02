"""Tests for clone mask extraction module."""

import numpy as np
import pytest
from ifish_tools.cloneextract.config import CloneExtractConfig, BoxCoords
from ifish_tools.cloneextract.core import (
    detect_mask_space,
    compute_relative_cbox,
    apply_morphological_closing,
    extract_clone_mask,
    process_brain,
)


class TestBoxCoords:
    """Test BoxCoords dataclass."""

    def test_from_matlab_1indexed(self):
        """MATLAB boxes are 1-indexed; convert to 0-indexed Python."""
        box = BoxCoords.from_matlab([560, 788, 775, 1210, 71, 420])
        assert box.ymin == 559
        assert box.ymax == 788
        assert box.xmin == 774
        assert box.xmax == 1210
        assert box.zmin == 70
        assert box.zmax == 420

    def test_from_python_0indexed(self):
        """Python boxes are already 0-indexed."""
        box = BoxCoords.from_python([559, 788, 774, 1210, 70, 420])
        assert box.ymin == 559
        assert box.ymax == 788

    def test_shape(self):
        box = BoxCoords.from_python([0, 10, 0, 20, 0, 30])
        assert box.shape() == (30, 10, 20)  # Z, Y, X

    def test_to_slices(self):
        box = BoxCoords.from_python([5, 15, 10, 30, 2, 8])
        zs, ys, xs = box.to_slices()
        assert zs == slice(2, 8)
        assert ys == slice(5, 15)
        assert xs == slice(10, 30)


class TestDetectMaskSpace:
    """Test auto-detection of mask coordinate space."""

    def test_whole_brain_size(self):
        """Mask matching neither bbox should be whole-brain."""
        mask = np.zeros((500, 1500, 1500), dtype=np.uint16)
        bbox = BoxCoords.from_python([33, 1117, 108, 1431, 1, 251])
        result = detect_mask_space(mask, bbox)
        assert result == "whole_brain"

    def test_bbox_size(self):
        """Mask matching bbox shape should be bbox-space."""
        bbox = BoxCoords.from_python([0, 100, 0, 200, 0, 50])
        mask = np.zeros(bbox.shape(), dtype=np.uint16)
        result = detect_mask_space(mask, bbox)
        assert result == "bbox"

    def test_unknown_size_raises(self):
        """Mask matching neither should raise ValueError."""
        bbox = BoxCoords.from_python([0, 100, 0, 200, 0, 50])
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        with pytest.raises(ValueError):
            detect_mask_space(mask, bbox)


class TestComputeRelativeCbox:
    """Test C-box coordinate conversion."""

    def test_relative_to_bbox(self):
        """C-box in bbox space = C-box - bbox_offset."""
        cbox = BoxCoords.from_python([560, 788, 775, 1210, 71, 200])
        bbox = BoxCoords.from_python([33, 1117, 108, 1431, 1, 251])
        rel = compute_relative_cbox(cbox, bbox)
        assert rel.ymin == 560 - 33
        assert rel.xmin == 775 - 108
        assert rel.zmin == 71 - 1

    def test_cbox_outside_bbox_raises(self):
        """C-box extending outside B-box should raise ValueError."""
        cbox = BoxCoords.from_python([0, 100, 0, 100, 0, 100])
        bbox = BoxCoords.from_python([50, 120, 50, 120, 50, 120])
        with pytest.raises(ValueError, match="outside"):
            compute_relative_cbox(cbox, bbox)


class TestMorphologicalClosing:
    """Test morphological closing."""

    def test_closing_fills_hole(self):
        """A small hole in a label should be filled by closing."""
        mask = np.zeros((20, 20, 20), dtype=np.uint16)
        mask[5:15, 5:15, 5:15] = 1
        mask[9, 9, 9] = 0  # small hole
        result = apply_morphological_closing(mask, radius=2)
        assert result[9, 9, 9] == 1

    def test_closing_preserves_labels(self):
        """Closing should not merge separate labels."""
        mask = np.zeros((30, 30, 30), dtype=np.uint16)
        mask[2:8, 2:8, 2:8] = 1
        mask[20:28, 20:28, 20:28] = 2
        result = apply_morphological_closing(mask, radius=2)
        labels = set(np.unique(result)) - {0}
        assert labels == {1, 2}

    def test_closing_radius_0_noop(self):
        """Radius 0 should return mask unchanged."""
        mask = np.zeros((10, 10, 10), dtype=np.uint16)
        mask[3:7, 3:7, 3:7] = 1
        result = apply_morphological_closing(mask, radius=0)
        np.testing.assert_array_equal(result, mask)


class TestExtractCloneMask:
    """Test clone extraction into B-box sized output."""

    def test_basic_extraction_whole_brain(self):
        """Extract clone from whole-brain mask into bbox output."""
        whole_mask = np.zeros((50, 50, 50), dtype=np.uint16)
        whole_mask[10:20, 10:20, 10:20] = 1

        cbox = BoxCoords.from_python([8, 25, 8, 25, 8, 25])
        bbox = BoxCoords.from_python([5, 40, 5, 40, 5, 40])

        result, stats = extract_clone_mask(
            whole_mask, cbox, bbox, mask_space="whole_brain",
            closing_radius=0
        )
        assert result.shape == bbox.shape()
        # cell at whole (10,10,10) → bbox-relative (5,5,5)
        assert result[5, 5, 5] == 1
        assert result[0, 0, 0] == 0
        assert stats["n_labels"] >= 1
        assert stats["n_voxels"] > 0

    def test_bbox_sized_input(self):
        """When mask is already bbox-sized."""
        bbox = BoxCoords.from_python([100, 200, 100, 200, 10, 60])
        bbox_mask = np.zeros(bbox.shape(), dtype=np.uint16)
        bbox_mask[10, 10, 10] = 5

        cbox = BoxCoords.from_python([105, 150, 105, 150, 15, 50])
        result, stats = extract_clone_mask(
            bbox_mask, cbox, bbox, mask_space="bbox",
            closing_radius=0
        )
        assert result.shape == bbox.shape()
        assert result[10, 10, 10] == 5
        assert stats["n_labels"] == 1
        assert stats["n_voxels"] == 1
