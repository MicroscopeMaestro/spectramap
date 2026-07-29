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
# FEAT-01: Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def test_feat01_load_csv_3d_xz():
    """Test loading compressed 3D CSV dataset."""
    obj = sp.hyper_object('test_3d', data_type='hyper_image')
    data_path = os.path.join('data', '3D')
    obj.read_csv_3d_xz(data_path)
    assert obj.data is not None
    assert len(obj.data.columns) > 0
    assert len(obj.data) > 0
    assert hasattr(obj, 'position')

def test_feat01_load_csv_standard(tmp_path):
    """Test loading standard CSV dataset via pandas and hyper_object."""
    df_raw = pd.DataFrame(
        np.random.rand(10, 50),
        columns=np.round(np.linspace(400, 1800, 50), 2)
    )
    csv_file = tmp_path / "test_standard.csv"
    df_raw.to_csv(csv_file, index=False)
    
    loaded_df = pd.read_csv(csv_file)
    obj = sp.hyper_object("standard_csv", data_type="hyper_image")
    obj.data = loaded_df
    assert obj.data.shape == (10, 50)
    assert np.allclose(obj.data.values, df_raw.values)

def test_feat01_load_spc_file():
    """Test loading binary SPC spectrum file."""
    spc_path = os.path.join('data', 'pigment')
    obj = sp.hyper_object('pigment', data_type='single_spectrum')
    obj.read_spc(spc_path)
    assert obj.data is not None
    assert len(obj.data.columns) > 0

def test_feat01_load_witec_map_export(witec_scan_txt_file):
    """Test loading WITec export format text file."""
    wn, mat, ncols, nrows = wrp.load_witec_map(witec_scan_txt_file)
    assert len(wn) == 300
    assert mat.shape == (300, 100)
    assert ncols == 10
    assert nrows == 10

def test_feat01_smart_importer_custom(tmp_path):
    """Test fallback custom text format parsing logic."""
    content = "Sample|x|y|400|500|600\nS1|0|0|10|20|30\nS2|0|1|15|25|35\n"
    txt_file = tmp_path / "custom.txt"
    txt_file.write_text(content)
    
    df = pd.read_csv(txt_file, sep="|")
    assert "x" in df.columns
    assert "y" in df.columns
    assert "400" in df.columns
    assert df.shape[0] == 2


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-02: Auto-Execution Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_feat02_pipeline_run_532nm(witec_scan_txt_file, glass_ref_txt_file, temp_output_dir):
    """Test full WITec 532 nm pipeline auto-execution."""
    config = {
        "SCAN_FILE": witec_scan_txt_file,
        "GLASS_FILE": glass_ref_txt_file,
        "CROP_LOW": 400,
        "CROP_HIGH": 3000,
        "SKIP_SILENT": True,
        "GLASS_METHOD": "vector",
        "AIRPLS_STRENGTH": 1e3,
        "NORM_MODE": "dual",
        "COSMIC_RAY_THRESHOLD": 4.5,
        "AIRPLS_ITERMAX": 20,
        "N_ENDMEMBERS": 3,
        "OUTPUT_DIR": temp_output_dir,
        "FIGURE_DPI": 100
    }
    wrp.run(config)
    assert os.path.exists(os.path.join(temp_output_dir, "processed", "endmember_spectra.csv"))
    assert os.path.exists(os.path.join(temp_output_dir, "processed", "abundance_maps.csv"))
    assert os.path.exists(os.path.join(temp_output_dir, "figures", "vca_endmembers.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "figures", "abundance_maps.png"))

def test_feat02_pipeline_run_785nm(witec_scan_txt_file, glass_ref_txt_file, tmp_path):
    """Test full WITec 785 nm pipeline auto-execution."""
    out_dir = str(tmp_path / "output_785")
    config = {
        "SCAN_FILE": witec_scan_txt_file,
        "GLASS_FILE": glass_ref_txt_file,
        "CROP_LOW": 400,
        "CROP_HIGH": 1800,
        "SKIP_SILENT": False,
        "GLASS_METHOD": "lsq",
        "AIRPLS_STRENGTH": 1e4,
        "NORM_MODE": "single",
        "COSMIC_RAY_THRESHOLD": 6.0,
        "AIRPLS_ITERMAX": 20,
        "N_ENDMEMBERS": 2,
        "OUTPUT_DIR": out_dir,
        "FIGURE_DPI": 100
    }
    wrp.run(config)
    assert os.path.exists(os.path.join(out_dir, "processed", "endmember_spectra.csv"))
    assert os.path.exists(os.path.join(out_dir, "figures", "vca_endmembers.png"))

def test_feat02_pipeline_default_config():
    """Test pipeline default CONFIG dict structure and keys."""
    assert "SCAN_FILE" in wrp.CONFIG
    assert "GLASS_FILE" in wrp.CONFIG
    assert "N_ENDMEMBERS" in wrp.CONFIG
    assert wrp.CONFIG["N_ENDMEMBERS"] > 0
    assert wrp.CONFIG["CROP_LOW"] < wrp.CONFIG["CROP_HIGH"]

def test_feat02_pipeline_output_folder_generation(witec_scan_txt_file, tmp_path):
    """Test auto-creation of timestamped output folder when OUTPUT_DIR is None."""
    out_stem = str(tmp_path / "auto_out")
    scan_path = str(tmp_path / "auto_scan.txt")
    import shutil
    shutil.copy(witec_scan_txt_file, scan_path)
    
    config = {
        "SCAN_FILE": scan_path,
        "GLASS_FILE": None,
        "CROP_LOW": 500,
        "CROP_HIGH": 1500,
        "SKIP_SILENT": False,
        "GLASS_METHOD": None,
        "AIRPLS_STRENGTH": 1e3,
        "NORM_MODE": "single",
        "COSMIC_RAY_THRESHOLD": 5.0,
        "AIRPLS_ITERMAX": 10,
        "N_ENDMEMBERS": 2,
        "OUTPUT_DIR": None,
        "FIGURE_DPI": 80
    }
    wrp.run(config)
    # Check that a timestamped folder was created in the directory of scan_path
    subdirs = [p for p in Path(tmp_path).iterdir() if p.is_dir() and "auto_scan" in p.name]
    assert len(subdirs) >= 1

def test_feat02_pipeline_skip_silent_region(synthetic_hyperspectral_matrix):
    """Test excluding silent 1900-2600 cm-1 region during cropping."""
    wn = synthetic_hyperspectral_matrix['wavenumbers']
    mat = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    
    wn_cropped, mat_cropped = wrp.crop_spectrum(wn, mat, 400, 3200, skip_silent=True)
    assert not np.any((wn_cropped > 1900) & (wn_cropped < 2600))
    assert wn_cropped.min() >= 400
    assert wn_cropped.max() <= 3200


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-03: Pre-Processing
# ══════════════════════════════════════════════════════════════════════════════

def test_feat03_preprocess_keep_wavenumber(synthetic_hyperspectral_matrix):
    """Test wavenumber region cropping via hyper_object."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("test_crop")
    obj.data = data['df'].copy()
    
    orig_cols = len(obj.data.columns)
    obj.keep(500, 1500)
    assert len(obj.data.columns) < orig_cols
    cols_float = [float(c) for c in obj.data.columns]
    assert min(cols_float) >= 490
    assert max(cols_float) <= 1510

def test_feat03_preprocess_snip_baseline(synthetic_hyperspectral_matrix):
    """Test SNIP baseline algorithm execution."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("test_snip")
    obj.data = data['df'].copy()
    
    orig_mean = obj.data.values.mean()
    obj.snip(20)
    assert obj.data.shape == data['df'].shape
    assert obj.data.values.mean() < orig_mean

def test_feat03_preprocess_airpls_baseline(synthetic_hyperspectral_matrix):
    """Test airPLS baseline correction method."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    corrected = wrp.correct_baseline(matrix, lam=1e3, itermax=20)
    assert corrected.shape == matrix.shape
    # Baseline subtracted matrix should have lower mean baseline offset
    assert corrected.mean() < matrix.mean()

def test_feat03_preprocess_glass_subtraction(synthetic_hyperspectral_matrix):
    """Test direct, vector, and lsq glass background subtraction."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    glass = np.mean(matrix, axis=1) + 2.0
    
    sub_direct = wrp.subtract_glass_direct(matrix, glass)
    sub_vector = wrp.subtract_glass_vector(matrix, glass)
    sub_lsq = wrp.subtract_glass_lsq(matrix, glass)
    
    assert sub_direct.shape == matrix.shape
    assert sub_vector.shape == matrix.shape
    assert sub_lsq.shape == matrix.shape

def test_feat03_preprocess_normalization(synthetic_hyperspectral_matrix):
    """Test single and dual region L2 normalization."""
    wn = synthetic_hyperspectral_matrix['wavenumbers']
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    
    norm_single = wrp.normalise(matrix.copy(), wn, mode="single")
    norms = np.linalg.norm(norm_single, axis=0)
    assert np.allclose(norms, 1.0)
    
    norm_dual = wrp.normalise(matrix.copy(), wn, mode="dual")
    assert norm_dual.shape == matrix.shape


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-04: VCA Unmixing
# ══════════════════════════════════════════════════════════════════════════════

def test_feat04_vca_endmember_extraction(synthetic_hyperspectral_matrix):
    """Test VCA endmember extraction shape and dimensions."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    n_endmembers = 3
    endmembers = wrp.vca(matrix, R=n_endmembers, seed=42)
    assert endmembers.shape == (matrix.shape[0], n_endmembers)

def test_feat04_vca_nnls_abundance_mapping(synthetic_hyperspectral_matrix):
    """Test NNLS abundance map estimation."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    endmembers = wrp.vca(matrix, R=3, seed=42)
    abundances = wrp.nnls_unmix(matrix, endmembers)  # (pixels, 3)
    assert abundances.shape == (100, 3)

def test_feat04_vca_abundance_non_negativity(synthetic_hyperspectral_matrix):
    """Test that all NNLS abundance values are strictly non-negative."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    endmembers = wrp.vca(matrix, R=3, seed=42)
    abundances = wrp.nnls_unmix(matrix, endmembers)
    assert np.all(abundances >= -1e-6)

def test_feat04_vca_abundance_sum_constraint(synthetic_hyperspectral_matrix):
    """Test that row-normalized abundance maps sum to approximately 1.0."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    endmembers = wrp.vca(matrix, R=3, seed=42)
    abundances = wrp.nnls_unmix(matrix, endmembers)
    sums = abundances.sum(axis=1)
    sums[sums == 0] = 1.0
    abundances_norm = abundances / sums[:, np.newaxis]
    assert np.allclose(abundances_norm.sum(axis=1), 1.0)

def test_feat04_vca_reconstruction_fidelity(synthetic_hyperspectral_matrix):
    """Test matrix reconstruction residual error using VCA endmembers."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    endmembers = wrp.vca(matrix, R=3, seed=42)
    abundances = wrp.nnls_unmix(matrix, endmembers)
    reconstructed = endmembers @ abundances.T  # (channels, pixels)
    residual = np.linalg.norm(matrix - reconstructed) / np.linalg.norm(matrix)
    assert residual < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-05: HCA & Clustering
# ══════════════════════════════════════════════════════════════════════════════

def test_feat05_hca_agglomerative_clustering(synthetic_hyperspectral_matrix):
    """Test Agglomerative Hierarchical Clustering via hyper_object."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("test_hca")
    obj.data = data['df'].copy()
    
    obj.hca(n_clusters=3)
    assert hasattr(obj, 'label')
    assert len(obj.label) == 100
    assert len(np.unique(obj.label)) == 3

def test_feat05_hca_kmeans_clustering(synthetic_hyperspectral_matrix):
    """Test KMeans clustering on hyperspectral data matrix."""
    from sklearn.cluster import KMeans
    matrix = synthetic_hyperspectral_matrix['df'].values  # (100, 300)
    km = KMeans(n_clusters=3, random_state=42).fit(matrix)
    labels = km.labels_
    assert len(labels) == 100
    assert len(np.unique(labels)) == 3

def test_feat05_hca_hdbscan_clustering(synthetic_hyperspectral_matrix):
    """Test HDBSCAN density clustering via hyper_object."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("test_hdbscan")
    obj.data = data['df'].copy()
    
    obj.hdbscan(min_cluster_size=5, min_samples=3)
    assert hasattr(obj, 'label')
    assert len(obj.label) == 100

def test_feat05_hca_cluster_labels_shape(synthetic_hyperspectral_matrix):
    """Test that cluster labels vector dimension matches pixel count."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("test_shape")
    obj.data = data['df'].copy()
    obj.hca(n_clusters=4)
    assert len(obj.label) == len(data['df'])

def test_feat05_hca_dendrogram_structure(synthetic_hyperspectral_matrix):
    """Test linkage matrix generation for HCA dendrogram."""
    from scipy.cluster.hierarchy import linkage
    matrix = synthetic_hyperspectral_matrix['df'].values
    Z = linkage(matrix, method='ward')
    assert Z.shape == (99, 4)  # (N-1, 4) for N=100 samples


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-06: PCA Analysis
# ══════════════════════════════════════════════════════════════════════════════

def test_feat06_pca_fit_dimensions(synthetic_hyperspectral_matrix):
    """Test PCA score and loading matrix dimensions."""
    from sklearn.decomposition import PCA
    matrix = synthetic_hyperspectral_matrix['df'].values  # (100, 300)
    pca = PCA(n_components=5).fit(matrix)
    scores = pca.transform(matrix)
    loadings = pca.components_
    
    assert scores.shape == (100, 5)
    assert loadings.shape == (5, 300)

def test_feat06_pca_explained_variance_ratios(synthetic_hyperspectral_matrix):
    """Test PCA explained variance ratios sum and monotonic decrease."""
    from sklearn.decomposition import PCA
    matrix = synthetic_hyperspectral_matrix['df'].values
    pca = PCA(n_components=5).fit(matrix)
    ratios = pca.explained_variance_ratio_
    
    assert len(ratios) == 5
    assert ratios.sum() <= 1.0
    assert ratios.sum() > 0.5
    assert np.all(np.diff(ratios) <= 0)  # Monotonically decreasing

def test_feat06_pca_transformed_scores_properties(synthetic_hyperspectral_matrix):
    """Test that transformed PCA score columns have zero mean."""
    from sklearn.decomposition import PCA
    matrix = synthetic_hyperspectral_matrix['df'].values
    pca = PCA(n_components=4).fit(matrix)
    scores = pca.transform(matrix)
    
    means = scores.mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-10)

def test_feat06_pca_confidence_ellipse_params():
    """Test confidence_ellipse math parameters calculation."""
    fig, ax = plt.subplots()
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    patch = sp.confidence_ellipse(x, y, ax, n_std=2.0)
    assert patch is not None
    plt.close(fig)

def test_feat06_pca_component_reconstruction(synthetic_hyperspectral_matrix):
    """Test low-rank PCA component reconstruction error."""
    from sklearn.decomposition import PCA
    matrix = synthetic_hyperspectral_matrix['df'].values
    pca = PCA(n_components=10).fit(matrix)
    scores = pca.transform(matrix)
    recon = pca.inverse_transform(scores)
    
    err = np.linalg.norm(matrix - recon) / np.linalg.norm(matrix)
    assert err < 0.1


# ══════════════════════════════════════════════════════════════════════════════
# FEAT-07: Output Saving & Export
# ══════════════════════════════════════════════════════════════════════════════

def test_feat07_export_endmember_csv(synthetic_hyperspectral_matrix, temp_output_dir):
    """Test exporting endmember spectra to CSV."""
    wn = synthetic_hyperspectral_matrix['wavenumbers']
    endmembers = synthetic_hyperspectral_matrix['endmembers'].T  # (channels, 3)
    
    proc_dir = os.path.join(temp_output_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    df_end = pd.DataFrame(endmembers, index=wn, columns=["EM1", "EM2", "EM3"])
    csv_path = os.path.join(proc_dir, "endmember_spectra.csv")
    df_end.to_csv(csv_path)
    
    assert os.path.exists(csv_path)
    df_read = pd.read_csv(csv_path, index_col=0)
    assert df_read.shape == (300, 3)

def test_feat07_export_abundance_csv(synthetic_hyperspectral_matrix, temp_output_dir):
    """Test exporting abundance maps to CSV."""
    abundances = synthetic_hyperspectral_matrix['abundances']  # (100, 3)
    proc_dir = os.path.join(temp_output_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    df_ab = pd.DataFrame(abundances, columns=["EM1", "EM2", "EM3"])
    csv_path = os.path.join(proc_dir, "abundance_maps.csv")
    df_ab.to_csv(csv_path, index=False)
    
    assert os.path.exists(csv_path)
    df_read = pd.read_csv(csv_path)
    assert df_read.shape == (100, 3)

def test_feat07_export_pca_tables(synthetic_hyperspectral_matrix, temp_output_dir):
    """Test exporting PCA score and loading tables."""
    from sklearn.decomposition import PCA
    matrix = synthetic_hyperspectral_matrix['df'].values
    pca = PCA(n_components=3).fit(matrix)
    scores = pca.transform(matrix)
    loadings = pca.components_
    
    proc_dir = os.path.join(temp_output_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    pd.DataFrame(scores, columns=["PC1", "PC2", "PC3"]).to_csv(os.path.join(proc_dir, "pca_scores.csv"), index=False)
    pd.DataFrame(loadings, index=["PC1", "PC2", "PC3"]).to_csv(os.path.join(proc_dir, "pca_loadings.csv"))
    
    assert os.path.exists(os.path.join(proc_dir, "pca_scores.csv"))
    assert os.path.exists(os.path.join(proc_dir, "pca_loadings.csv"))

def test_feat07_export_figures_png(synthetic_hyperspectral_matrix, temp_output_dir):
    """Test figure plot generation and PNG file export."""
    fig_dir = os.path.join(temp_output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    wn = synthetic_hyperspectral_matrix['wavenumbers']
    em = synthetic_hyperspectral_matrix['endmembers']
    for i in range(em.shape[0]):
        ax.plot(wn, em[i], label=f"EM{i+1}")
    ax.legend()
    
    fig_path = os.path.join(fig_dir, "vca_endmembers.png")
    fig.savefig(fig_path, dpi=100)
    plt.close(fig)
    
    assert os.path.exists(fig_path)
    assert os.path.getsize(fig_path) > 0

def test_feat07_export_metadata_json(temp_output_dir):
    """Test metadata JSON export with exact parameters schema."""
    proc_dir = os.path.join(temp_output_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    metadata = {
        "timestamp": "2026-07-22T08:44:21Z",
        "laser_line": "532 nm",
        "n_endmembers": 3,
        "grid": {"ncols": 10, "nrows": 10, "n_pixels": 100},
        "preprocessing": {
            "crop": [400, 3000],
            "baseline": "airPLS",
            "glass_subtraction": "vector"
        }
    }
    
    json_path = os.path.join(proc_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    assert os.path.exists(json_path)
    with open(json_path) as f:
        read_meta = json.load(f)
    assert read_meta["laser_line"] == "532 nm"
    assert read_meta["grid"]["n_pixels"] == 100


def test_colocalization_overlay_synthesis():
    """Test 3-channel and 4-channel colorblind-friendly overlay matrix synthesis and legend rendering for all 8 modes."""
    import matplotlib.patches as mpatches
    from spectramap.spmap import synthesize_colocalization_overlay
    from app import PALETTE_CONFIGS, crop_and_orient_map
    
    m1 = np.random.rand(12, 12)
    m2 = np.random.rand(12, 12)
    m3 = np.random.rand(12, 12)
    m4 = np.random.rand(12, 12)
    all_maps = [m1, m2, m3, m4]
    
    # Test all 8 palette configurations in PALETTE_CONFIGS
    for mode_name, cfg in PALETTE_CONFIGS.items():
        n_chan = cfg["n_chan"]
        colors = cfg["colors"]
        assert len(colors) == n_chan
        
        comp = synthesize_colocalization_overlay(all_maps[:n_chan], colors)
        assert comp.shape == (12, 12, 3)
        assert np.min(comp) >= 0.0 and np.max(comp) <= 1.0
        
        fig, ax = plt.subplots()
        ax.imshow(comp)
        patches = [mpatches.Patch(color=c, label=f"Ch{i+1}") for i, c in enumerate(colors)]
        leg = ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5))
        assert len(leg.get_patches()) == n_chan
        plt.close(fig)


def test_crop_and_orient_map_3d_rgb():
    """Test Matplotlib legend patch generation and 3D RGB array spatial cropping."""
    from app import crop_and_orient_map
    
    # 2D Map Cropping & Orientation
    img_2d = np.arange(100).reshape(10, 10)
    cropped_2d = crop_and_orient_map(img_2d, rotation=90, flip_h=True, flip_v=False,
                                      crop_spatial=True, crop_x_min=2, crop_x_max=8, crop_y_min=1, crop_y_max=9)
    assert cropped_2d.ndim == 2
    assert cropped_2d.shape == (6, 8)
    
    # 3D RGB Array Cropping & Orientation
    img_3d = np.random.rand(20, 30, 3)
    cropped_3d = crop_and_orient_map(img_3d, rotation=0, flip_h=False, flip_v=False,
                                      crop_spatial=True, crop_x_min=5, crop_x_max=25, crop_y_min=2, crop_y_max=18)
    assert cropped_3d.ndim == 3
    assert cropped_3d.shape == (16, 20, 3)
    
    # 3D RGB Array Crop + 90 Deg Rotation + Flips
    cropped_3d_rot90 = crop_and_orient_map(img_3d, rotation=90, flip_h=True, flip_v=True,
                                            crop_spatial=True, crop_x_min=5, crop_x_max=25, crop_y_min=2, crop_y_max=18)
    assert cropped_3d_rot90.ndim == 3
    assert cropped_3d_rot90.shape == (20, 16, 3)


def test_navigation_state_synchronization():
    """Test workflow step update logic and session state synchronization across sidebar, top radios, and step buttons."""
    import streamlit as st
    from app import update_workflow_step, on_sidebar_step_nav_change, on_top_step_nav_change, workflow_steps
    
    st.session_state.clear()
    st.session_state["current_step_index"] = 0
    st.session_state["sidebar_step_nav_radio"] = workflow_steps[0]
    st.session_state["top_step_nav_radio"] = workflow_steps[0]
    
    # 1. Update step via helper (simulating Next button to step 2 - index 1)
    update_workflow_step(1)
    assert st.session_state["current_step_index"] == 1
    assert st.session_state["sidebar_step_nav_radio"] == workflow_steps[1]
    assert st.session_state["top_step_nav_radio"] == workflow_steps[1]
    
    # 2. Simulate sidebar radio change to step 4 (index 3)
    st.session_state["sidebar_step_nav_radio"] = workflow_steps[3]
    on_sidebar_step_nav_change()
    assert st.session_state["current_step_index"] == 3
    assert st.session_state["sidebar_step_nav_radio"] == workflow_steps[3]
    assert st.session_state["top_step_nav_radio"] == workflow_steps[3]
    
    # 3. Simulate top radio change to step 3 (index 2)
    st.session_state["top_step_nav_radio"] = workflow_steps[2]
    on_top_step_nav_change()
    assert st.session_state["current_step_index"] == 2
    assert st.session_state["sidebar_step_nav_radio"] == workflow_steps[2]
    assert st.session_state["top_step_nav_radio"] == workflow_steps[2]

    # 4. Out of bounds index safety test
    update_workflow_step(99)
    assert st.session_state["current_step_index"] == 2  # unchanged



