"""Test module for puncta registration functions."""

import pytest
import numpy as np
from pathlib import Path
from ifish_tools.countmatrix.regis_puncta import (
    load_affine_transform,
    transform_points_forward,
    load_displacement_field,
    resize_displacement_field,
    apply_demons_displacement,
    load_bbox,
    get_cropped_image_size,
)


def test_transform_points_forward_identity():
    """Test that identity transform doesn't change points."""
    T = np.eye(4)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    z = np.array([7.0, 8.0, 9.0])
    
    x_out, y_out, z_out = transform_points_forward(T, x, y, z)
    
    np.testing.assert_array_almost_equal(x_out, x)
    np.testing.assert_array_almost_equal(y_out, y)
    np.testing.assert_array_almost_equal(z_out, z)


def test_apply_demons_displacement_zero_field():
    """Test that zero displacement field doesn't change puncta."""
    # Create test puncta (1-based coordinates)
    puncta = np.array([
        [10.0, 20.0, 5.0, 100.0],
        [15.0, 25.0, 6.0, 150.0],
    ])
    
    # Zero displacement field (Ny=50, Nx=60, Nz=10, 3 components)
    Dfull = np.zeros((50, 60, 10, 3))
    
    result = apply_demons_displacement(puncta, Dfull)
    
    # Should return same coordinates (within numerical precision)
    np.testing.assert_array_almost_equal(result[:, :3], puncta[:, :3], decimal=5)
