"""Tests for countmatrix.qc module."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ifish_tools.countmatrix.qc import summarize_clone


class TestSummarizeClone:
    """Test summarize_clone function."""

    def test_summarize_clone_basic(self):
        """Test summarize_clone with basic synthetic data."""
        import anndata as ad

        # Create synthetic count matrix
        # 10 cells x 5 genes
        counts = np.array([
            [10, 5, 0, 2, 3],   # cell 0: 20 total, 4 genes
            [0, 0, 0, 0, 0],     # cell 1: 0 total, 0 genes (zero-count)
            [5, 10, 15, 0, 0],   # cell 2: 30 total, 3 genes
            [1, 1, 1, 1, 1],     # cell 3: 5 total, 5 genes
            [20, 0, 0, 0, 0],    # cell 4: 20 total, 1 gene
            [3, 3, 3, 3, 3],     # cell 5: 15 total, 5 genes
            [0, 5, 5, 5, 0],     # cell 6: 15 total, 3 genes
            [10, 10, 10, 10, 10],# cell 7: 50 total, 5 genes
            [2, 2, 2, 2, 2],     # cell 8: 10 total, 5 genes
            [0, 0, 0, 0, 5],     # cell 9: 5 total, 1 gene
        ])

        # Create AnnData
        adata = ad.AnnData(
            X=sp.csr_matrix(counts),
            obs=pd.DataFrame({
                "brain_id": ["08"] * 10,
                "clone_id": ["1"] * 10,
            }, index=[f"cell{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"gene{i}" for i in range(5)])
        )

        # Run summarize_clone
        summary = summarize_clone(adata, "brain08_clone1")

        # Check results
        assert summary["clone"] == "brain08_clone1"
        assert summary["n_cells"] == 10
        assert summary["n_genes"] == 5
        assert summary["total_counts"] == 170
        assert summary["median_counts_per_cell"] == 15.0  # median of sorted [0, 5, 5, 10, 15, 15, 20, 20, 30, 50]
        assert summary["mean_counts_per_cell"] == 17.0
        assert summary["median_genes_per_cell"] == 3.5  # median of [4, 0, 3, 5, 1, 5, 3, 5, 5, 1]
        assert summary["mean_genes_per_cell"] == 3.2
        assert summary["zero_count_cells"] == 1  # cell 1
        assert summary["zero_count_genes"] == 0  # all genes have >0 counts

    def test_summarize_clone_all_zeros(self):
        """Test summarize_clone with all-zero matrix."""
        import anndata as ad

        # Create all-zero count matrix
        counts = np.zeros((5, 3), dtype=int)

        adata = ad.AnnData(
            X=sp.csr_matrix(counts),
            obs=pd.DataFrame({
                "brain_id": ["08"] * 5,
                "clone_id": ["1"] * 5,
            }, index=[f"cell{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"gene{i}" for i in range(3)])
        )

        summary = summarize_clone(adata, "test_clone")

        assert summary["clone"] == "test_clone"
        assert summary["n_cells"] == 5
        assert summary["n_genes"] == 3
        assert summary["total_counts"] == 0
        assert summary["median_counts_per_cell"] == 0.0
        assert summary["mean_counts_per_cell"] == 0.0
        assert summary["median_genes_per_cell"] == 0.0
        assert summary["mean_genes_per_cell"] == 0.0
        assert summary["zero_count_cells"] == 5
        assert summary["zero_count_genes"] == 3

    def test_summarize_clone_single_cell(self):
        """Test summarize_clone with single cell."""
        import anndata as ad

        counts = np.array([[10, 0, 5, 0, 3]])  # 1 cell, 5 genes

        adata = ad.AnnData(
            X=sp.csr_matrix(counts),
            obs=pd.DataFrame({
                "brain_id": ["08"],
                "clone_id": ["1"],
            }, index=["cell0"]),
            var=pd.DataFrame(index=[f"gene{i}" for i in range(5)])
        )

        summary = summarize_clone(adata, "single_cell")

        assert summary["clone"] == "single_cell"
        assert summary["n_cells"] == 1
        assert summary["n_genes"] == 5
        assert summary["total_counts"] == 18
        assert summary["median_counts_per_cell"] == 18.0
        assert summary["mean_counts_per_cell"] == 18.0
        assert summary["median_genes_per_cell"] == 3.0
        assert summary["mean_genes_per_cell"] == 3.0
        assert summary["zero_count_cells"] == 0
        assert summary["zero_count_genes"] == 2  # genes 1 and 3
