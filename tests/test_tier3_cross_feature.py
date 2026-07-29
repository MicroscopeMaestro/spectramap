import pytest
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from spectramap import spmap as sp
import witec_raman_pipeline as wrp


def test_cross_vca_plus_hca(synthetic_hyperspectral_matrix):
    """Test interaction: VCA endmember unmixing followed by HCA clustering on abundance maps."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    endmembers = wrp.vca(matrix, R=3, seed=42)
    abundances = wrp.nnls_unmix(matrix, endmembers)  # (pixels, 3)
    
    # Create hyper_object for abundance maps and run HCA
    obj_ab = sp.hyper_object("vca_abundance_hca")
    obj_ab.data = pd.DataFrame(abundances, columns=["EM1", "EM2", "EM3"])
    obj_ab.hca(n_clusters=3)
    
    assert hasattr(obj_ab, 'label')
    assert len(obj_ab.label) == 100
    assert len(np.unique(obj_ab.label)) == 3

def test_cross_glass_subtraction_plus_airpls_plus_norm(synthetic_hyperspectral_matrix):
    """Test interaction: Glass subtraction -> airPLS baseline correction -> Dual normalization."""
    wn = synthetic_hyperspectral_matrix['wavenumbers']
    matrix = synthetic_hyperspectral_matrix['df'].values.T  # (channels, pixels)
    glass = np.mean(matrix, axis=1) + 1.0
    
    # 1. Glass subtraction
    mat_sub = wrp.subtract_glass_vector(matrix, glass)
    # 2. airPLS baseline
    mat_base = wrp.correct_baseline(mat_sub, lam=1e3, itermax=10)
    # 3. Dual normalization
    mat_norm = wrp.normalise(mat_base, wn, mode="dual")
    
    assert mat_norm.shape == matrix.shape
    assert not np.isnan(mat_norm).any()

def test_cross_cosmic_ray_plus_snip_plus_pca(synthetic_hyperspectral_matrix):
    """Test interaction: Cosmic ray removal -> SNIP baseline -> PCA dimensional reduction."""
    from sklearn.decomposition import PCA
    matrix = synthetic_hyperspectral_matrix['df'].values.T
    matrix = matrix.copy()
    
    # Add fake cosmic ray spikes
    matrix[50, 12] = 500.0
    matrix[120, 45] = 600.0
    
    # 1. Cosmic ray removal
    clean_mat, n_fixed = wrp.remove_cosmic_rays(matrix, nrows=10, ncols=10, threshold=4.5)
    assert n_fixed >= 2
    
    # 2. SNIP baseline via hyper_object
    obj = sp.hyper_object("snip_pca")
    obj.data = pd.DataFrame(clean_mat.T, columns=np.round(synthetic_hyperspectral_matrix['wavenumbers'], 2))
    obj.snip(20)
    
    # 3. PCA
    pca = PCA(n_components=3).fit(obj.data.values)
    scores = pca.transform(obj.data.values)
    
    assert scores.shape == (100, 3)

def test_cross_pipeline_end_to_end_full_flow(witec_scan_txt_file, glass_ref_txt_file, temp_output_dir):
    """Test interaction: End-to-end WITec pipeline flow execution."""
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
        "AIRPLS_ITERMAX": 15,
        "N_ENDMEMBERS": 3,
        "OUTPUT_DIR": temp_output_dir,
        "FIGURE_DPI": 80
    }
    wrp.run(config)
    
    # Verify outputs across both processed tables and figures
    proc_files = os.listdir(os.path.join(temp_output_dir, "processed"))
    fig_files = os.listdir(os.path.join(temp_output_dir, "figures"))
    
    assert "endmember_spectra.csv" in proc_files
    assert "abundance_maps.csv" in proc_files
    assert "metadata.json" in proc_files
    assert "vca_endmembers.png" in fig_files
    assert "abundance_maps.png" in fig_files

def test_cross_smart_importer_to_hyper_object_to_vca(tmp_path):
    """Test interaction: Custom text data parsing -> hyper_object conversion -> VCA unmixing."""
    # Build synthetic custom delimited text dataset
    wns = np.linspace(400, 1800, 50)
    lines = ["Sample|x|y|" + "|".join(f"{w:.1f}" for w in wns)]
    for y in range(5):
        for x in range(5):
            vals = np.random.rand(50) + 2.0 * np.exp(-((wns - 1000) ** 2) / (2 * 50 ** 2))
            lines.append(f"S_{x}_{y}|{x}|{y}|" + "|".join(f"{v:.4f}" for v in vals))
            
    custom_txt = tmp_path / "custom_importer.txt"
    custom_txt.write_text("\n".join(lines))
    
    df = pd.read_csv(custom_txt, sep="|")
    obj = sp.hyper_object("imported", data_type="hyper_image")
    obj.data = df.drop(columns=["Sample", "x", "y"])
    obj.position = df[["x", "y"]]
    
    # Run VCA on imported data
    matrix = obj.data.values.T  # (50 channels, 25 pixels)
    endmembers = wrp.vca(matrix, R=2, seed=42)
    abundances = wrp.nnls_unmix(matrix, endmembers)
    
    assert endmembers.shape == (50, 2)
    assert abundances.shape == (25, 2)

def test_cross_preprocessing_pipeline_consistency(synthetic_hyperspectral_matrix):
    """Test consistency between hyper_object methods and pipeline functions."""
    data = synthetic_hyperspectral_matrix
    obj = sp.hyper_object("consistency")
    obj.data = data['df'].copy()
    
    # Run SNIP on hyper_object
    obj.snip(15)
    
    # Run SNIP directly via spmap function
    base = sp.snip(data['df'].copy(), 15)
    direct_sub = data['df'].copy() - base
    
    assert np.allclose(obj.data.values, direct_sub.values)

def test_cross_spc_load_to_hca_to_pca():
    """Test interaction: SPC loading -> HCA clustering -> PCA on cluster means."""
    from sklearn.decomposition import PCA
    spc_path = os.path.join('data', 'pigment')
    obj = sp.hyper_object('pigment', data_type='single_spectrum')
    obj.read_spc(spc_path)
    
    df = obj.data
    assert df is not None
    # Duplicate rows to simulate multiple spectra for clustering & PCA
    df_multi = pd.concat([df] * 10, ignore_index=True)
    obj_multi = sp.hyper_object('pigment_multi', data_type='multi_spectra')
    obj_multi.data = df_multi
    
    obj_multi.hca(n_clusters=2)
    pca = PCA(n_components=2).fit(df_multi.values)
    scores = pca.transform(df_multi.values)
    
    assert len(obj_multi.label) == 10
    assert scores.shape == (10, 2)

def test_cross_multithread_or_parallel_airpls(synthetic_hyperspectral_matrix):
    """Test interaction: Single-thread vs joblib parallel airPLS output consistency."""
    matrix = synthetic_hyperspectral_matrix['df'].values.T[:, :20]  # 20 pixels
    
    # Run sequential airPLS
    seq_baselines = [wrp._airpls_single(matrix[:, i], lam=1e3, itermax=10) for i in range(20)]
    seq_corrected = matrix - np.array(seq_baselines).T
    
    # Run pipeline correct_baseline
    pipe_corrected = wrp.correct_baseline(matrix, lam=1e3, itermax=10)
    
    assert np.allclose(seq_corrected, pipe_corrected)
