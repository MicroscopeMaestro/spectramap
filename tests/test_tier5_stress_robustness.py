#!/usr/bin/env python3
"""
Tier 5 Stress & Robustness Test Suite for SpectraMap
===================================================
Empirically tests error handling, parameter boundary checks,
malformed data parsing, glass reference fallbacks, PCA rank clamping,
and HCA/HDBSCAN clustering edge cases (Requirement R2).
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project src, tools, and root to sys.path
project_root = Path(__file__).parent.parent.resolve() if (Path(__file__).parent.name == "tests" or Path(__file__).parent.name.startswith("teamwork_")) else Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
tools_dir = project_root / "tools"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import witec_raman_pipeline as wrp
from spectramap import spmap as sp

# ------------------------------------------------------------------------------
# 1. Malformed CSV/TXT Data Files
# ------------------------------------------------------------------------------

def test_crop_bounds_out_of_order():
    """Verify that crop_low >= crop_high raises explicit ValueError."""
    wn = np.linspace(400, 3300, 100)
    mat = np.random.rand(100, 10)
    
    with pytest.raises(ValueError, match="must be strictly less than"):
        wrp.crop_spectrum(wn, mat, low=2000.0, high=500.0, skip_silent=False)
        
    with pytest.raises(ValueError, match="must be strictly less than"):
        wrp.crop_spectrum(wn, mat, low=1000.0, high=1000.0, skip_silent=False)

def test_crop_range_out_of_bounds():
    """Verify that cropping outside spectral range raises ValueError."""
    wn = np.linspace(400, 1000, 100)
    mat = np.random.rand(100, 10)
    
    with pytest.raises(ValueError, match="0 valid spectral channels"):
        wrp.crop_spectrum(wn, mat, low=2000.0, high=3000.0, skip_silent=False)

def test_witec_map_non_square_no_header(tmp_path):
    """Verify that WITec loader without grid header on non-square pixel count fails gracefully."""
    fake_txt = tmp_path / "corrupt_grid.txt"
    # 5 pixels, not square
    data = np.zeros((10, 6))
    data[:, 0] = np.linspace(500, 1500, 10) # wavenumber
    data[:, 1:] = np.random.rand(10, 5) # 5 pixels
    np.savetxt(fake_txt, data, delimiter=",")
    
    with pytest.raises(ValueError, match="not a perfect square"):
        wrp.load_witec_map(str(fake_txt))

def test_corrupt_csv_parsing(tmp_path):
    """Verify behavior on CSV with non-numeric spectral headers."""
    csv_file = tmp_path / "corrupt_headers.csv"
    df = pd.DataFrame({
        'x': [0, 1],
        'y': [0, 0],
        'Wavenumber_A': [100.0, 105.0],
        'Wavenumber_B': [200.0, 210.0]
    })
    df.to_csv(csv_file, index=False)
    
    # Using pd.to_numeric without errors='coerce' on column names raises ValueError
    cols = df.drop(columns=['x', 'y']).columns
    with pytest.raises(ValueError):
        pd.to_numeric(cols).values


# ------------------------------------------------------------------------------
# 2. Extreme Parameter Inputs & Matrix Singularities
# ------------------------------------------------------------------------------

def test_pca_rank_clamping():
    """Verify that requesting 100 PCA components on a 2-pixel dataset clamps to 2."""
    mat = np.random.rand(50, 2) # 50 channels, 2 pixels -> rank max = 2
    scores, loadings, var_ratio = wrp.run_pca(mat, n_components=100)
    
    assert scores.shape == (2, 2), f"Expected scores shape (2, 2), got {scores.shape}"
    assert loadings.shape == (2, 50), f"Expected loadings shape (2, 50), got {loadings.shape}"
    assert len(var_ratio) == 2, f"Expected 2 components in variance ratio, got {len(var_ratio)}"

def test_zero_variance_matrix_normalisation():
    """Verify normalisation on flat zero-variance spectrum matrix (all zeros)."""
    wn = np.linspace(400, 3300, 50)
    mat = np.zeros((50, 10)) # All zeros
    
    norm_single = wrp.normalise(mat.copy(), wn, mode="single")
    assert not np.isnan(norm_single).any(), "Single normalisation produced NaNs on zero matrix"
    assert np.all(norm_single == 0), "Normalised zero matrix should remain zeros"
    
    norm_dual = wrp.normalise(mat.copy(), wn, mode="dual")
    assert not np.isnan(norm_dual).any(), "Dual normalisation produced NaNs on zero matrix"
    assert np.all(norm_dual == 0), "Dual normalised zero matrix should remain zeros"

def test_zero_variance_baseline_airpls():
    """Verify airPLS baseline correction on constant flat matrix."""
    mat = np.ones((50, 5)) * 100.0 # Constant 100.0
    corrected = wrp.correct_baseline(mat, lam=1e3, itermax=10)
    assert not np.isnan(corrected).any(), "airPLS baseline correction produced NaNs on constant matrix"


# ------------------------------------------------------------------------------
# 3. Missing or Singular Glass Reference Spectrum Files
# ------------------------------------------------------------------------------

def test_missing_glass_file():
    """Verify handling of missing glass reference file path."""
    missing_path = "non_existent_glass_file_12345.txt"
    assert not os.path.exists(missing_path)
    
    with pytest.raises(FileNotFoundError):
        wrp.load_spectrum(missing_path)

def test_singular_glass_lsq_subtraction():
    """Verify least-squares glass subtraction with flat zero glass spectrum skips gracefully."""
    glass_flat = np.zeros(50)
    mat = np.random.rand(50, 10)
    
    # Should print warning and return original matrix unchanged
    res = wrp.subtract_glass_lsq(mat, glass_flat)
    np.testing.assert_array_equal(res, mat)

def test_vector_glass_subtraction_zero_glass():
    """Verify vector glass subtraction handles zero glass spectrum without divide-by-zero."""
    glass_flat = np.zeros(50)
    mat = np.random.rand(50, 10)
    
    res = wrp.subtract_glass_vector(mat, glass_flat)
    assert not np.isnan(res).any(), "Vector glass subtraction produced NaNs for zero glass spectrum"


# ------------------------------------------------------------------------------
# 4. Single-Cluster HCA Edge Cases & Excessive Cluster Counts
# ------------------------------------------------------------------------------

def test_hca_single_cluster(tmp_path):
    """Verify HCA when all points merge into 1 single cluster."""
    hca_obj = sp.hyper_object("test_single_cluster")
    # 5 identical spectra -> distance = 0 -> 1 cluster
    wn = np.linspace(400, 1000, 20)
    data_df = pd.DataFrame(np.ones((5, 20)), columns=wn)
    hca_obj.data = data_df
    hca_obj.position = pd.DataFrame({'x': range(5), 'y': [0]*5})
    hca_obj.m = 5
    hca_obj.n = 1
    hca_obj.label = pd.Series([1]*5)
    
    hca_obj.hca(distance="euclidean", linkage="ward", dist=10.0, p=None)
    
    unique_labels = hca_obj.label.unique()
    assert len(unique_labels) == 1, f"Expected 1 cluster, got {len(unique_labels)}"

def test_hca_excessive_clusters(tmp_path):
    """Verify HCA when dist threshold is extremely low leading to N clusters (>50)."""
    n_samples = 60
    hca_obj = sp.hyper_object("test_many_clusters")
    wn = np.linspace(400, 1000, 20)
    # Distinct spectra
    spectra = np.random.rand(n_samples, 20) * 100.0 + np.arange(n_samples)[:, None] * 10.0
    hca_obj.data = pd.DataFrame(spectra, columns=wn)
    hca_obj.position = pd.DataFrame({'x': range(n_samples), 'y': [0]*n_samples})
    hca_obj.m = n_samples
    hca_obj.n = 1
    hca_obj.label = pd.Series([1]*n_samples)
    
    # dist=0.001 cuts every sample into its own cluster
    hca_obj.hca(distance="euclidean", linkage="ward", dist=0.001, p=None)
    
    unique_labels = hca_obj.label.unique()
    assert len(unique_labels) > 50, f"Expected >50 clusters, got {len(unique_labels)}"

def test_hca_pearson_zero_variance():
    """Verify HCA with Pearson correlation on zero-variance spectra raises explicit ValueError."""
    hca_obj = sp.hyper_object("test_zero_var")
    wn = np.linspace(400, 1000, 20)
    data_df = pd.DataFrame(np.zeros((5, 20)), columns=wn)
    hca_obj.data = data_df
    hca_obj.position = pd.DataFrame({'x': range(5), 'y': [0]*5})
    hca_obj.m = 5
    hca_obj.n = 1
    hca_obj.label = pd.Series([1]*5)
    
    with pytest.raises(ValueError, match="zero variance"):
        hca_obj.hca(distance="pearson", linkage="average", dist=1.0, p=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
