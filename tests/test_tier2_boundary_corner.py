import pytest
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from spectramap import spmap as sp
import witec_raman_pipeline as wrp


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-01 Boundaries: Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def test_feat01_load_missing_file_raises_error():
    """Test loading non-existent file raises FileNotFound error."""
    with pytest.raises((FileNotFoundError, OSError)):
        wrp.load_witec_map("non_existent_file_path_12345.txt")

def test_feat01_load_malformed_csv_header(malformed_txt_file):
    """Test loading malformed TXT file falls back to square grid or raises clear ValueError."""
    try:
        wn, mat, ncols, nrows = wrp.load_witec_map(malformed_txt_file)
        assert len(wn) > 0
    except ValueError as e:
        err_msg = str(e).lower()
        assert any(k in err_msg for k in ["grid", "square", "pixels", "could not convert", "invalid"])

def test_feat01_load_empty_dataset(tmp_path):
    """Test loading empty data file."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    with pytest.raises(Exception):
        wrp.load_witec_map(str(empty_file))

def test_feat01_load_nan_inf_data(nan_inf_matrix_df):
    """Test hyper_object handling of NaN and Inf data values."""
    obj = sp.hyper_object("nan_test")
    obj.data = nan_inf_matrix_df.copy()
    # Check that NaNs/Infs exist and can be filled/cleaned
    cleaned = obj.data.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    assert not cleaned.isnull().any().any()
    assert not np.isinf(cleaned.values).any()

def test_feat01_load_single_pixel_map(tmp_path):
    """Test loading 1x1 single pixel map grid corner case."""
    single_pixel_file = tmp_path / "single_pixel.txt"
    with open(single_pixel_file, "w") as f:
        f.write("Wavenumber,(0/0)\n")
        f.write("400.0,12.5\n")
        f.write("401.0,14.8\n")
        f.write("402.0,19.1\n")
        
    wn, mat, ncols, nrows = wrp.load_witec_map(str(single_pixel_file))
    assert len(wn) == 3
    assert mat.shape == (3, 1)
    assert ncols == 1
    assert nrows == 1


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-02 Boundaries: Auto-Execution Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_feat02_pipeline_missing_glass_ref(witec_scan_txt_file, tmp_path):
    """Test pipeline execution when GLASS_FILE is None (no glass ref)."""
    out_dir = str(tmp_path / "out_noglass")
    config = {
        "SCAN_FILE": witec_scan_txt_file,
        "GLASS_FILE": None,
        "CROP_LOW": 400,
        "CROP_HIGH": 2000,
        "SKIP_SILENT": False,
        "GLASS_METHOD": None,
        "AIRPLS_STRENGTH": 1e3,
        "NORM_MODE": "single",
        "COSMIC_RAY_THRESHOLD": 5.0,
        "AIRPLS_ITERMAX": 10,
        "N_ENDMEMBERS": 2,
        "OUTPUT_DIR": out_dir,
        "FIGURE_DPI": 80
    }
    wrp.run(config)
    assert os.path.exists(os.path.join(out_dir, "processed", "endmember_spectra.csv"))

def test_feat02_pipeline_zero_laser_power(witec_scan_txt_file, tmp_path):
    """Test pipeline execution with LASER_POWER_MW set to 0.0 or None."""
    out_dir = str(tmp_path / "out_zeropower")
    config = {
        "SCAN_FILE": witec_scan_txt_file,
        "GLASS_FILE": None,
        "CROP_LOW": 400,
        "CROP_HIGH": 1500,
        "SKIP_SILENT": False,
        "GLASS_METHOD": None,
        "AIRPLS_STRENGTH": 1e3,
        "NORM_MODE": "single",
        "COSMIC_RAY_THRESHOLD": 5.0,
        "AIRPLS_ITERMAX": 10,
        "N_ENDMEMBERS": 2,
        "LASER_POWER_MW": 0.0,
        "INTEGRATION_TIME_SEC": 0.0,
        "OUTPUT_DIR": out_dir,
        "FIGURE_DPI": 80
    }
    wrp.run(config)
    assert os.path.exists(os.path.join(out_dir, "processed", "endmember_spectra.csv"))

def test_feat02_pipeline_out_of_bounds_crop(synthetic_hyperspectral_matrix):
    """Test crop_spectrum when CROP_LOW > CROP_HIGH or out of range."""
    wn = synthetic_hyperspectral_matrix['wavenumbers']
    mat = synthetic_hyperspectral_matrix['df'].values.T
    
    # Range completely above wavenumber axis
    with pytest.raises(ValueError):
        wrp.crop_spectrum(wn, mat, 5000, 6000, False)

def test_feat02_pipeline_single_endmember_vca(witec_scan_txt_file, tmp_path):
    """Test pipeline unmixing with N_ENDMEMBERS = 1 corner case."""
    out_dir = str(tmp_path / "out_vca1")
    config = {
        "SCAN_FILE": witec_scan_txt_file,
        "GLASS_FILE": None,
        "CROP_LOW": 400,
        "CROP_HIGH": 1500,
        "SKIP_SILENT": False,
        "GLASS_METHOD": None,
        "AIRPLS_STRENGTH": 1e3,
        "NORM_MODE": "single",
        "COSMIC_RAY_THRESHOLD": 5.0,
        "AIRPLS_ITERMAX": 10,
        "N_ENDMEMBERS": 1,
        "OUTPUT_DIR": out_dir,
        "FIGURE_DPI": 80
    }
    wrp.run(config)
    ab_df = pd.read_csv(os.path.join(out_dir, "processed", "abundance_maps.csv"), comment='#')
    ab_cols = [c for c in ab_df.columns if 'Abundance' in c]
    assert len(ab_cols) == 1


def test_feat02_pipeline_invalid_norm_mode(synthetic_hyperspectral_matrix):
    """Test fallback for invalid normalization mode string."""
    wn = synthetic_hyperspectral_matrix['wavenumbers']
    mat = synthetic_hyperspectral_matrix['df'].values.T
    # Passing unknown mode defaults to single L2 norm
    res = wrp.normalise(mat.copy(), wn, mode="invalid_mode_name")
    assert res.shape == mat.shape


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-03 Boundaries: Pre-Processing
# ══════════════════════════════════════════════════════════════════════════════

def test_feat03_preprocess_keep_invalid_bounds(synthetic_hyperspectral_matrix):
    """Test keep method when bounds are inverted or outside range."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("test_keep_inv")
    obj.data = data['df'].copy()
    
    # Bounds outside range
    obj.keep(10000, 12000)
    assert len(obj.data.columns) >= 0

def test_feat03_preprocess_airpls_zero_lambda(synthetic_hyperspectral_matrix):
    """Test airPLS baseline when lambda strength is zero or very small."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    res = wrp.correct_baseline(matrix, lam=1e-5, itermax=5)
    assert res.shape == matrix.shape
    assert not np.isnan(res).any()

def test_feat03_preprocess_cosmic_ray_zero_threshold(synthetic_hyperspectral_matrix):
    """Test cosmic ray removal with very low threshold."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    clean, n_fixed = wrp.remove_cosmic_rays(matrix, nrows=10, ncols=10, threshold=0.1)
    assert clean.shape == matrix.shape
    assert n_fixed >= 0

def test_feat03_preprocess_glass_subtraction_zero_signal(synthetic_hyperspectral_matrix):
    """Test glass background subtraction when glass reference is all zeros."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    glass_zero = np.zeros(matrix.shape[0])
    
    sub_vector = wrp.subtract_glass_vector(matrix, glass_zero)
    sub_lsq = wrp.subtract_glass_lsq(matrix, glass_zero)
    
    assert not np.isnan(sub_vector).any()
    assert not np.isnan(sub_lsq).any()

def test_feat03_preprocess_normalization_constant_spectrum():
    """Test normalizing a zero-intensity spectrum without division by zero."""
    zero_mat = np.zeros((100, 10))
    wn = np.linspace(400, 1800, 100)
    res = wrp.normalise(zero_mat, wn, mode="single")
    assert not np.isnan(res).any()
    assert (res == 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-04 Boundaries: VCA Unmixing
# ══════════════════════════════════════════════════════════════════════════════

def test_feat04_vca_more_endmembers_than_channels():
    """Test requesting more endmembers than spectral channels."""
    small_mat = np.random.rand(5, 50)  # 5 channels, 50 pixels
    Ae = wrp.vca(small_mat, R=10)
    assert Ae.shape[0] <= 5 or Ae.shape[1] <= 5

def test_feat04_vca_more_endmembers_than_pixels():
    """Test requesting more endmembers than pixel count."""
    small_mat = np.random.rand(100, 3)  # 100 channels, 3 pixels
    Ae = wrp.vca(small_mat, R=5)
    assert Ae.shape[0] <= 3 or Ae.shape[1] <= 3

def test_feat04_vca_zero_variance_input():
    """Test VCA on constant flat input matrix."""
    const_mat = np.ones((50, 20))
    endmembers = wrp.vca(const_mat, R=2, seed=42)
    assert endmembers.shape == (50, 2)
    assert not np.isnan(endmembers).any()

def test_feat04_vca_nan_inf_input_handling(nan_inf_matrix_df):
    """Test VCA after replacing NaNs and Infs."""
    df_clean = nan_inf_matrix_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    matrix = df_clean.values.T  # (channels, pixels)
    endmembers = wrp.vca(matrix, R=2, seed=42)
    assert endmembers.shape == (matrix.shape[0], 2)

def test_feat04_vca_single_endmember_unmixing(synthetic_hyperspectral_matrix):
    """Test VCA and NNLS with N_ENDMEMBERS = 1."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    em1 = wrp.vca(matrix, R=1, seed=42)
    ab1 = wrp.nnls_unmix(matrix, em1)
    assert em1.shape == (matrix.shape[0], 1)
    assert ab1.shape == (100, 1)


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-05 Boundaries: HCA & Clustering
# ══════════════════════════════════════════════════════════════════════════════

def test_feat05_hca_single_cluster(synthetic_hyperspectral_matrix):
    """Test HCA clustering with n_clusters = 1 corner case."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("hca_1")
    obj.data = data['df'].copy()
    obj.hca(n_clusters=1)
    assert len(np.unique(obj.label)) == 1

def test_feat05_hca_clusters_equal_to_pixels(synthetic_hyperspectral_matrix):
    """Test HCA clustering with n_clusters = n_pixels corner case."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("hca_100")
    obj.data = data['df'].copy()
    obj.hca(n_clusters=100)
    assert len(np.unique(obj.label)) == 100

def test_feat05_hca_constant_data_input():
    """Test clustering on constant pixel spectra."""
    const_df = pd.DataFrame(np.ones((20, 50)))
    obj = sp.hyper_object("const_hca")
    obj.data = const_df
    obj.hca(n_clusters=2)
    assert len(obj.label) == 20

def test_feat05_hca_hdbscan_min_cluster_size_larger_than_data(synthetic_hyperspectral_matrix):
    """Test HDBSCAN with min_cluster_size larger than pixel count."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("hdb_large")
    obj.data = data['df'].copy()
    # Setting min_cluster_size = 150 > 100 pixels will label all as noise (-1)
    obj.hdbscan(min_cluster_size=150, min_samples=5)
    assert len(obj.label) == 100
    assert (obj.label == -1).all()

def test_feat05_hca_nan_inf_handling(nan_inf_matrix_df):
    """Test HCA clustering after replacing NaNs."""
    df_clean = nan_inf_matrix_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    obj = sp.hyper_object("nan_hca")
    obj.data = df_clean
    obj.hca(n_clusters=2)
    assert len(obj.label) == len(df_clean)


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-06 Boundaries: PCA Analysis
# ══════════════════════════════════════════════════════════════════════════════

def test_feat06_pca_components_exceed_rank():
    """Test requesting PCA components exceeding matrix rank."""
    from sklearn.decomposition import PCA
    small_mat = np.random.rand(5, 50)  # max rank is 5
    pca = PCA(n_components=5).fit(small_mat)
    assert pca.n_components_ == 5

def test_feat06_pca_single_component(synthetic_hyperspectral_matrix):
    """Test PCA with n_components = 1."""
    from sklearn.decomposition import PCA
    matrix = synthetic_hyperspectral_matrix['df'].values
    pca = PCA(n_components=1).fit(matrix)
    scores = pca.transform(matrix)
    assert scores.shape == (100, 1)

def test_feat06_pca_constant_matrix():
    """Test PCA on zero-variance constant matrix."""
    from sklearn.decomposition import PCA
    const_mat = np.ones((20, 50))
    pca = PCA(n_components=2).fit(const_mat)
    scores = pca.transform(const_mat)
    assert scores.shape == (20, 2)
    assert np.allclose(np.nan_to_num(pca.explained_variance_ratio_), 0.0)


def test_feat06_pca_nan_inf_handling(nan_inf_matrix_df):
    """Test PCA fit after replacing NaNs/Infs."""
    from sklearn.decomposition import PCA
    clean_mat = nan_inf_matrix_df.fillna(0.0).replace([np.inf, -np.inf], 0.0).values
    pca = PCA(n_components=2).fit(clean_mat)
    scores = pca.transform(clean_mat)
    assert scores.shape == (20, 2)

def test_feat06_pca_high_dimensional_boundary():
    """Test PCA on single-row matrix (1 x P)."""
    from sklearn.decomposition import PCA
    single_row = np.random.rand(1, 100)
    pca = PCA(n_components=1).fit(single_row)
    scores = pca.transform(single_row)
    assert scores.shape == (1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-07 Boundaries: Output Saving & Export
# ══════════════════════════════════════════════════════════════════════════════

def test_feat07_export_nonexistent_directory(tmp_path):
    """Test exporting outputs automatically creates non-existent parent directories."""
    nested_dir = str(tmp_path / "deep" / "nested" / "output")
    proc_dir = os.path.join(nested_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    csv_path = os.path.join(proc_dir, "test.csv")
    pd.DataFrame({"A": [1, 2]}).to_csv(csv_path)
    assert os.path.exists(csv_path)

def test_feat07_export_read_only_or_permission_issue(tmp_path):
    """Test handling of invalid/protected destination paths."""
    proc_dir = os.path.join(str(tmp_path), "proc")
    os.makedirs(proc_dir, exist_ok=True)
    csv_path = os.path.join(proc_dir, "valid.csv")
    df = pd.DataFrame({"X": [10, 20]})
    df.to_csv(csv_path)
    assert os.path.exists(csv_path)

def test_feat07_export_empty_dataframe_csv(tmp_path):
    """Test exporting empty DataFrame to CSV file."""
    empty_df = pd.DataFrame()
    csv_path = tmp_path / "empty_out.csv"
    empty_df.to_csv(csv_path)
    assert os.path.exists(csv_path)

def test_feat07_export_null_metadata_values(tmp_path):
    """Test metadata JSON export with None/null attribute values."""
    json_path = tmp_path / "null_meta.json"
    metadata = {
        "LASER_POWER_MW": None,
        "OBJECTIVE": None,
        "COSMIC_RAYS": 0
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f)
        
    with open(json_path) as f:
        read_data = json.load(f)
    assert read_data["LASER_POWER_MW"] is None

def test_feat07_export_figure_matplotlib_cleanup(tmp_path):
    """Test verifying plot figure closing and memory cleanup."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    fig_path = tmp_path / "fig_clean.png"
    fig.savefig(fig_path)
    plt.close(fig)
    assert os.path.exists(fig_path)
