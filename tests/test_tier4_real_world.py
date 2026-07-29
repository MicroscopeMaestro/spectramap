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


def test_realworld_532nm_witec_hyperspectral_map(witec_scan_txt_file, glass_ref_txt_file, temp_output_dir):
    """Real-World Scenario 1: 532 nm WITec hyperspectral mapping of biological/polymer sample (fingerprint + C-H stretch)."""
    config = {
        "SCAN_FILE": witec_scan_txt_file,
        "GLASS_FILE": glass_ref_txt_file,
        "CROP_LOW": 400,
        "CROP_HIGH": 3200,
        "SKIP_SILENT": True,
        "GLASS_METHOD": "vector",
        "AIRPLS_STRENGTH": 1e3,
        "NORM_MODE": "dual",
        "COSMIC_RAY_THRESHOLD": 4.5,
        "AIRPLS_ITERMAX": 25,
        "N_ENDMEMBERS": 3,
        "OUTPUT_DIR": temp_output_dir,
        "FIGURE_DPI": 100
    }
    wrp.run(config)
    
    # Verify generated output structure and content validity
    end_df = pd.read_csv(os.path.join(temp_output_dir, "processed", "endmember_spectra.csv"), comment='#', index_col=0)
    ab_df = pd.read_csv(os.path.join(temp_output_dir, "processed", "abundance_maps.csv"), comment='#')
    
    assert end_df.shape[1] == 3
    assert ab_df.shape[0] == 100
    assert not ab_df.isnull().any().any()
    assert (ab_df.values >= -1e-5).all()


def test_realworld_785nm_witec_hyperspectral_map(tmp_path):
    """Real-World Scenario 2: 785 nm WITec Raman scan (fingerprint region only, LSQ glass subtraction)."""
    np.random.seed(123)
    wns = np.linspace(400, 1950, 250)
    n_pixels = 64  # 8x8 grid
    
    # 2 pure components (e.g. Titanium Dioxide & Organic dye)
    c1 = np.exp(-((wns - 610) ** 2) / (2 * 15 ** 2)) + 0.8 * np.exp(-((wns - 1430) ** 2) / (2 * 20 ** 2))
    c2 = np.exp(-((wns - 850) ** 2) / (2 * 15 ** 2)) + 0.9 * np.exp(-((wns - 1600) ** 2) / (2 * 25 ** 2))
    
    mix = np.random.dirichlet((1, 1), size=n_pixels) @ np.vstack([c1, c2])  # 64 x 250
    glass = 2.0 + 8.0 * np.exp(-((wns - 1200) ** 2) / (2 * 300 ** 2))
    
    scan_file = tmp_path / "witec_785_scan.txt"
    bg_file = tmp_path / "witec_785_bg.txt"
    
    headers = ["Wavenumber"] + [f"({x}/{y})" for y in range(8) for x in range(8)]
    with open(scan_file, "w") as f:
        f.write(",".join(headers) + "\n")
        for i, wn in enumerate(wns):
            vals = mix[:, i] + glass[i] * 0.5 + np.random.normal(0, 0.01, size=n_pixels)
            f.write(f"{wn:.2f}," + ",".join(f"{v:.4f}" for v in vals) + "\n")
            
    with open(bg_file, "w") as f:
        f.write("Wavenumber,Background\n")
        for wn, g in zip(wns, glass):
            f.write(f"{wn:.2f},{g:.4f}\n")
            
    out_dir = str(tmp_path / "out_785_rw")
    config = {
        "SCAN_FILE": str(scan_file),
        "GLASS_FILE": str(bg_file),
        "CROP_LOW": 450,
        "CROP_HIGH": 1900,
        "SKIP_SILENT": False,
        "GLASS_METHOD": "lsq",
        "AIRPLS_STRENGTH": 1e5,
        "NORM_MODE": "single",
        "COSMIC_RAY_THRESHOLD": 6.0,
        "AIRPLS_ITERMAX": 20,
        "N_ENDMEMBERS": 2,
        "OUTPUT_DIR": out_dir,
        "FIGURE_DPI": 80
    }
    wrp.run(config)
    
    assert os.path.exists(os.path.join(out_dir, "processed", "endmember_spectra.csv"))
    assert os.path.exists(os.path.join(out_dir, "figures", "vca_endmembers.png"))

def test_realworld_3d_compressed_dataset_analysis():
    """Real-World Scenario 3: Real compressed 3D.csv.xz dataset loading, preprocessing, and HCA."""
    obj = sp.hyper_object('3D_rw', data_type='hyper_image')
    obj.read_csv_3d_xz(os.path.join('data', '3D'))
    
    assert obj.data is not None
    assert len(obj.data) > 0
    
    # Crop to fingerprint region
    obj.keep(400, 1800)
    obj.snip(15)
    obj.hca(n_clusters=3)
    
    assert hasattr(obj, 'label')
    assert len(obj.label) == len(obj.data)
    assert len(np.unique(obj.label)) == 3

def test_realworld_paracetamol_sample_analysis():
    """Real-World Scenario 4: Real paracetaminol.csv dataset loading & preprocessing."""
    para_path = os.path.join('data', 'paracetaminol.csv')
    df = pd.read_csv(para_path)
    
    obj = sp.hyper_object('paracetamol', data_type='multi_spectra')
    obj.data = df
    
    assert len(obj.data) > 0
    obj.keep(400, 1800)
    obj.snip(10)
    
    assert obj.data.shape[1] < df.shape[1]

def test_realworld_messy_dataset_importer():
    """Real-World Scenario 5: Real messy_dataset.txt custom text file importing."""
    messy_path = os.path.join('data', 'messy_dataset.txt')
    assert os.path.exists(messy_path)
    
    with open(messy_path, "r") as f:
        lines = f.readlines()
        
    # Header starts at line index 6 (skip 6 metadata header lines)
    data_lines = lines[6:]
    df = pd.read_csv(Path(messy_path), sep="|", skiprows=6)
    
    assert "sample_id" in df.columns or "x_coord" in df.columns
    df_renamed = df.rename(columns={"sample_id": "label", "x_coord": "x", "y_coord": "y"})
    
    obj = sp.hyper_object("messy", data_type="multi_spectra")
    obj.data = df_renamed.drop(columns=["label", "x", "y"], errors="ignore")
    obj.label = df_renamed["label"] if "label" in df_renamed else pd.Series([1]*len(df))
    
    assert len(obj.data) == 6
    assert len(obj.label) == 6

def test_realworld_microplastics_multi_component_unmixing(tmp_path):
    """Real-World Scenario 6: 5-component microplastics mixture (PET, PMMA, PE, PS, PVC) unmixing."""
    np.random.seed(999)
    wns = np.linspace(400, 3100, 400)
    n_pixels = 81  # 9x9 grid
    
    # 5 polymer endmember peak profiles
    em_pet  = np.exp(-((wns - 1610) ** 2) / (2 * 15 ** 2)) + 0.8 * np.exp(-((wns - 1720) ** 2) / (2 * 15 ** 2))
    em_pmma = np.exp(-((wns - 810)  ** 2) / (2 * 15 ** 2)) + 0.9 * np.exp(-((wns - 1730) ** 2) / (2 * 15 ** 2))
    em_pe   = np.exp(-((wns - 1060) ** 2) / (2 * 15 ** 2)) + 1.0 * np.exp(-((wns - 2880) ** 2) / (2 * 25 ** 2))
    em_ps   = np.exp(-((wns - 1000) ** 2) / (2 * 15 ** 2)) + 0.9 * np.exp(-((wns - 3050) ** 2) / (2 * 20 ** 2))
    em_pvc  = np.exp(-((wns - 635)  ** 2) / (2 * 15 ** 2)) + 0.7 * np.exp(-((wns - 1425) ** 2) / (2 * 20 ** 2))
    
    E = np.vstack([em_pet, em_pmma, em_pe, em_ps, em_pvc])  # 5 x 400
    A = np.random.dirichlet((1, 1, 1, 1, 1), size=n_pixels)  # 81 x 5
    
    matrix = (A @ E).T  # 400 x 81
    
    # Perform 5-endmember VCA unmixing
    extracted_EM = wrp.vca(matrix, R=5, seed=42)
    abundances = wrp.nnls_unmix(matrix, extracted_EM)
    
    assert extracted_EM.shape == (400, 5)
    assert abundances.shape == (81, 5)
    assert not np.isnan(abundances).any()
    
    # Export microplastics analysis table
    out_dir = tmp_path / "microplastics_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(abundances, columns=["PET", "PMMA", "PE", "PS", "PVC"]).to_csv(out_dir / "microplastics_abundances.csv", index=False)
    assert (out_dir / "microplastics_abundances.csv").exists()
