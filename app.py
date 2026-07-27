import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json
import copy

import matplotlib.patches as mpatches

# Add tools and src to sys.path
current_dir = Path(__file__).parent
tools_dir = str(current_dir / "tools")
src_dir = str(current_dir / "src")
if tools_dir not in sys.path:
    sys.path.append(tools_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import witec_raman_pipeline as wrp
import importlib
importlib.reload(wrp)

from spectramap import spmap as sp
try:
    from smart_importer import parse_with_ollama
except ImportError:
    parse_with_ollama = None

st.set_page_config(page_title="SpectraMap GUI", layout="wide")

st.title("SpectraMap GUI — Hyperspectral Raman Analysis")

# Helper functions
def get_data_files():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(data_dir):
        files = glob.glob(os.path.join(data_dir, '*.csv.xz')) + glob.glob(os.path.join(data_dir, '*.spc')) + glob.glob(os.path.join(data_dir, '*.csv'))
        return sorted([os.path.basename(f) for f in files if not os.path.basename(f).startswith('.')])
    return []

def orient_map(img_2d: np.ndarray, rotation: int = 0, flip_h: bool = False, flip_v: bool = False) -> np.ndarray:
    """Applies rotation (0, 90, 180, 270 degrees) and horizontal/vertical flips to a 2D spatial map image."""
    out = np.array(img_2d, copy=True)
    if rotation == 90:
        out = np.rot90(out, 1)
    elif rotation == 180:
        out = np.rot90(out, 2)
    elif rotation == 270:
        out = np.rot90(out, 3)
    if flip_h:
        out = np.fliplr(out)
    if flip_v:
        out = np.flipud(out)
    return out

def crop_and_orient_map(img_2d: np.ndarray, 
                        rotation: int = 0, 
                        flip_h: bool = False, 
                        flip_v: bool = False,
                        crop_spatial: bool = False,
                        crop_x_min: int = 0,
                        crop_x_max: int = None,
                        crop_y_min: int = 0,
                        crop_y_max: int = None) -> np.ndarray:
    """Crops and rotates/flips a 2D or 3D (RGB) spatial map image."""
    out = np.array(img_2d, copy=True)
    if crop_spatial and out.ndim in (2, 3):
        h, w = out.shape[:2]
        x0 = max(0, min(int(crop_x_min), w - 1))
        x1 = max(x0 + 1, min(int(crop_x_max) if crop_x_max is not None else w, w))
        y0 = max(0, min(int(crop_y_min), h - 1))
        y1 = max(y0 + 1, min(int(crop_y_max) if crop_y_max is not None else h, h))
        out = out[y0:y1, x0:x1] if out.ndim == 2 else out[y0:y1, x0:x1, :]
    return orient_map(out, rotation=rotation, flip_h=flip_h, flip_v=flip_v)

def synthesize_colocalization_overlay(channel_maps: list, channel_colors: list) -> np.ndarray:
    """
    Synthesizes a 3-channel RGB composite image from 3 or 4 spatial component maps
    and their corresponding RGB color vectors.
    """
    if not channel_maps or not channel_colors:
        raise ValueError("channel_maps and channel_colors must not be empty.")
    
    H, W = channel_maps[0].shape[:2]
    r_comp = np.zeros((H, W), dtype=float)
    g_comp = np.zeros((H, W), dtype=float)
    b_comp = np.zeros((H, W), dtype=float)
    
    for m, (r, g, b) in zip(channel_maps, channel_colors):
        ptp = np.ptp(m)
        norm_m = (m - np.min(m)) / (ptp if ptp != 0 else 1.0)
        r_comp += norm_m * r
        g_comp += norm_m * g
        b_comp += norm_m * b
        
    comp = np.stack([np.clip(r_comp, 0.0, 1.0), np.clip(g_comp, 0.0, 1.0), np.clip(b_comp, 0.0, 1.0)], axis=-1)
    return comp

def get_safe_index(prev_val, options_list, default_idx):
    if prev_val in options_list:
        return options_list.index(prev_val)
    if not options_list:
        return 0
    return min(default_idx, max(0, len(options_list) - 1))

def sync_endmember_selection(trigger_key: str):
    new_sel = st.session_state.get(trigger_key, [])
    st.session_state["global_selected_endmembers"] = new_sel
    for key in ["step2_vca_select", "step3_vca_grid_select", "step3_vca_overlap_select", "tab1_vca_select", "tab2_vca_select_grid", "tab2_vca_select_overlap"]:
        if key != trigger_key and key in st.session_state:
            st.session_state[key] = new_sel

def sync_pc_selection(trigger_key: str):
    new_sel = st.session_state.get(trigger_key, [])
    st.session_state["global_selected_pcs"] = new_sel
    for key in ["step3_pca_select", "step4_pca_select", "tab1_pca_select", "tab2_pca_select"]:
        if key != trigger_key and key in st.session_state:
            st.session_state[key] = new_sel

def save_uploaded_file(uploaded_file, filename):
    temp_dir = Path(__file__).parent / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)

def render_folder_browser(start_dir: str):
    """Interactive directory browser widget for selecting folders in Streamlit"""
    curr = st.session_state.get("browser_curr_dir", start_dir)
    curr_path = Path(curr).resolve()
    
    if not curr_path.exists():
        curr_path = Path.cwd()
        
    st.session_state["browser_curr_dir"] = str(curr_path)
    st.markdown(f"📁 **Current Folder:** `{curr_path}`")
    
    col_nav1, col_nav2 = st.columns([1, 2])
    with col_nav1:
        if curr_path.parent != curr_path:
            if st.button("⬆️ Up Level", key="btn_dir_up", use_container_width=True):
                st.session_state["browser_curr_dir"] = str(curr_path.parent)
                st.rerun()
                
    subdirs = []
    try:
        subdirs = [d.name for d in curr_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except Exception:
        pass
        
    with col_nav2:
        if subdirs:
            selected_sub = st.selectbox("Subdirectories", ["-- Select subdirectory --"] + sorted(subdirs), key="sb_subdirs")
            if selected_sub != "-- Select subdirectory --":
                st.session_state["browser_curr_dir"] = str(curr_path / selected_sub)
                st.rerun()
        else:
            st.caption("No subdirectories found.")
            
    return str(curr_path)

def load_saved_files_from_upload(up_ab, up_em):
    """Load abundance_maps.csv and endmember_spectra.csv directly from browser file uploaders"""
    try:
        df_ab = pd.read_csv(up_ab, comment="#")
        df_em = pd.read_csv(up_em, comment="#")
        df_ab = df_ab.rename(columns={"X_pixel": "x", "Y_pixel": "y", "X": "x", "Y": "y", "X_pos": "x", "Y_pos": "y"})
        
        wn = df_em.iloc[:, 0].values
        Ae = df_em.iloc[:, 1:].values
        n_em = Ae.shape[1]
        
        ncols = int(df_ab["x"].max() + 1) if "x" in df_ab.columns else 10
        nrows = int(df_ab["y"].max() + 1) if "y" in df_ab.columns else 10
        ab_cols = [c for c in df_ab.columns if c not in ["x", "y"]]
        ab_3d = np.zeros((nrows, ncols, len(ab_cols)))
        for _, r in df_ab.iterrows():
            xi, yi = int(r["x"]), int(r["y"])
            if 0 <= xi < ncols and 0 <= yi < nrows:
                ab_3d[yi, xi, :] = r[ab_cols].values
                
        res = {
            "out_root": "./upload_export",
            "proc_root": "./upload_export",
            "fig_root": "./upload_export",
            "data_name": "Uploaded Export Files",
            "analysis_method": "VCA (Unmixing)",
            "skip_silent": True,
            "map_interpolation": "nearest",
            "metadata": {},
            "run_use_glass": False,
            "glass_wn": None,
            "glass_int": None,
            "abundances": ab_3d,
            "ncols": ncols,
            "nrows": nrows,
            "position": df_ab[["x", "y"]] if "x" in df_ab.columns and "y" in df_ab.columns else pd.DataFrame({"x": [0], "y": [0]}),
            "wavenumber": wn,
            "Ae": Ae,
            "n_endmembers": n_em,
            "parsed_labels": [c for c in df_em.columns if c != df_em.columns[0]],
            "df_endmembers": df_em,
            "df_abundances": df_ab,
        }
        st.session_state.pipeline_results = res
        st.session_state.pipeline_success = True
        return True
    except Exception as e:
        st.error(f"Error loading uploaded files: {e}")
        return False

def load_saved_results(saved_dir_path: str):
    """Load previously saved export results directory into st.session_state.pipeline_results"""
    dir_path = Path(saved_dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        st.error(f"Saved results directory not found: '{saved_dir_path}'")
        return False
        
    meta_path = dir_path / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            pass
            
    proc_dir = dir_path / "processed_data"
    fig_dir = dir_path / "figures"
    
    analysis_method = meta.get("pipeline_analysis", "VCA (Unmixing)")
    data_name = meta.get("dataset_name", dir_path.name)
    
    res = {
        "out_root": str(dir_path),
        "proc_root": str(proc_dir if proc_dir.exists() else dir_path),
        "fig_root": str(fig_dir if fig_dir.exists() else dir_path),
        "data_name": data_name,
        "analysis_method": analysis_method,
        "skip_silent": meta.get("skip_silent", True),
        "map_interpolation": meta.get("map_interpolation", "nearest"),
        "metadata": meta,
        "run_use_glass": meta.get("use_glass", False),
        "glass_wn": None,
        "glass_int": None,
    }
    
    endmembers_csv = proc_dir / "vca_endmembers.csv"
    if not endmembers_csv.exists(): endmembers_csv = proc_dir / "endmember_spectra.csv"
    if not endmembers_csv.exists(): endmembers_csv = dir_path / "endmember_spectra.csv"
    if not endmembers_csv.exists(): endmembers_csv = dir_path / "vca_endmembers.csv"
    
    abundances_csv = proc_dir / "vca_abundances.csv"
    if not abundances_csv.exists(): abundances_csv = proc_dir / "abundance_maps.csv"
    if not abundances_csv.exists(): abundances_csv = dir_path / "abundance_maps.csv"
    if not abundances_csv.exists(): abundances_csv = dir_path / "vca_abundances.csv"
    
    pca_scores_csv = proc_dir / "pca_scores.csv"
    if not pca_scores_csv.exists(): pca_scores_csv = dir_path / "pca_scores.csv"
    
    pca_loadings_csv = proc_dir / "pca_loadings.csv"
    if not pca_loadings_csv.exists(): pca_loadings_csv = dir_path / "pca_loadings.csv"
    
    if endmembers_csv.exists() and abundances_csv.exists():
        df_em = pd.read_csv(endmembers_csv, comment="#")
        df_ab = pd.read_csv(abundances_csv, comment="#")
        df_ab = df_ab.rename(columns={"X_pixel": "x", "Y_pixel": "y", "X": "x", "Y": "y", "X_pos": "x", "Y_pos": "y"})
        
        wn = df_em.iloc[:, 0].values
        Ae = df_em.iloc[:, 1:].values
        n_em = Ae.shape[1]
        
        ncols, nrows = 10, 10
        if "x" in df_ab.columns and "y" in df_ab.columns:
            ncols = int(df_ab["x"].max() + 1)
            nrows = int(df_ab["y"].max() + 1)
            ab_cols = [c for c in df_ab.columns if c not in ["x", "y"]]
            ab_3d = np.zeros((nrows, ncols, len(ab_cols)))
            for _, r in df_ab.iterrows():
                xi, yi = int(r["x"]), int(r["y"])
                if 0 <= xi < ncols and 0 <= yi < nrows:
                    ab_3d[yi, xi, :] = r[ab_cols].values
            res.update({
                "abundances": ab_3d,
                "ncols": ncols,
                "nrows": nrows,
                "position": df_ab[["x", "y"]],
            })
        else:
            res.update({
                "abundances": np.zeros((10, 10, n_em)),
                "ncols": 10,
                "nrows": 10,
                "position": pd.DataFrame({"x": [0], "y": [0]}),
            })
            
        res.update({
            "wavenumber": wn,
            "Ae": Ae,
            "n_endmembers": n_em,
            "parsed_labels": meta.get("endmember_labels") or [c for c in df_em.columns if c != df_em.columns[0]],
            "df_endmembers": df_em,
            "df_abundances": df_ab,
        })

    if pca_scores_csv.exists() and pca_loadings_csv.exists():
        df_scores = pd.read_csv(pca_scores_csv, comment="#")
        df_loadings = pd.read_csv(pca_loadings_csv, comment="#", index_col=0)
        
        wn = pd.to_numeric(df_loadings.columns).values
        loadings = df_loadings.values
        scores_cols = [c for c in df_scores.columns if c not in ["x", "y"]]
        scores = df_scores[scores_cols].values.T
        pca_comp = scores.shape[0]
        
        ncols = int(df_scores["x"].max() + 1) if "x" in df_scores.columns else 10
        nrows = int(df_scores["y"].max() + 1) if "y" in df_scores.columns else 10
        
        res.update({
            "wavenumber": wn,
            "pca_scores": scores,
            "pca_loadings": loadings,
            "pca_components": pca_comp,
            "ncols": ncols,
            "nrows": nrows,
            "position": df_scores[["x", "y"]] if "x" in df_scores.columns and "y" in df_scores.columns else pd.DataFrame({'x': [0], 'y': [0]}),
            "df_pca_scores": df_scores,
            "df_pca_loadings": df_loadings,
        })
        
    st.session_state.pipeline_results = res
    st.session_state.pipeline_success = True
    return True

# ==============================================================================
# SIDEBAR REORGANIZATION INTO 5 COLLAPSIBLE EXPANDERS
# ==============================================================================
st.sidebar.markdown("### 🎛️ SpectraMap Control Panel")

# Container 1: Data Input & Selection (expanded=True)
with st.sidebar.expander("📂 Data Input & Selection", expanded=True):
    dataset_source = st.radio("Dataset Source", ["Sample Datasets", "Upload Custom File", "Local Scan File (.txt)", "📂 Saved Results Directory", "Smart Importer (AI)"], index=0)
    
    selected_sample = None
    uploaded_file = None
    local_scan_path = None
    
    if dataset_source == "Sample Datasets":
        data_files = get_data_files()
        default_idx = 0
        if "bladder.csv.xz" in data_files:
            default_idx = data_files.index("bladder.csv.xz")
        if data_files:
            selected_sample = st.selectbox("Select Sample Dataset", data_files, index=default_idx)
        else:
            st.error("No sample datasets found in 'data/' directory.")
    elif dataset_source == "Upload Custom File":
        uploaded_file = st.file_uploader("Upload Data File (.csv, .csv.xz, .txt, .spc)", type=["csv", "xz", "txt", "spc"])
    elif dataset_source == "Local Scan File (.txt)":
        local_scan_path = st.text_input("Local File Path (.txt)", value=st.session_state.get("local_scan_path_val", ""))
    elif dataset_source == "📂 Saved Results Directory":
        saved_mode = st.radio("Selection Mode", ["📁 Interactive Folder Browser", "📤 Upload Export CSVs", "⌨️ Direct Path Entry"], index=0, key="saved_sel_mode")
        
        if saved_mode == "📁 Interactive Folder Browser":
            default_saved_dir = r"c:\Users\Juan\Documents\GitHub\spectramap\data\processed_data\processed_data\sample 2"
            selected_dir = render_folder_browser(default_saved_dir)
            if st.button("📂 Load Selected Folder", use_container_width=True, key="btn_load_browser_folder") or st.session_state.get("auto_load_dir") != selected_dir:
                if load_saved_results(selected_dir):
                    st.session_state["auto_load_dir"] = selected_dir
                    st.sidebar.success(f"Loaded saved results from: '{selected_dir}'")
        elif saved_mode == "📤 Upload Export CSVs":
            up_ab = st.file_uploader("Upload Abundance Maps CSV (abundance_maps.csv or vca_abundances.csv)", type=["csv"], key="up_ab_file")
            up_em = st.file_uploader("Upload Endmember Spectra CSV (endmember_spectra.csv or vca_endmembers.csv)", type=["csv"], key="up_em_file")
            if up_ab is not None and up_em is not None:
                if st.button("📂 Load Uploaded Files", use_container_width=True, key="btn_load_up_csvs"):
                    if load_saved_files_from_upload(up_ab, up_em):
                        st.sidebar.success("Successfully loaded uploaded abundance maps and endmember spectra!")
        else:
            default_saved_dir = r"c:\Users\Juan\Documents\GitHub\spectramap\data\processed_data\processed_data\sample 2"
            if not Path(default_saved_dir).exists():
                default_saved_dir = "./export_results"
            saved_dir_input = st.text_input("Saved Results Directory Path", value=default_saved_dir)
            if st.button("📂 Load Saved Results", use_container_width=True):
                if load_saved_results(saved_dir_input):
                    st.sidebar.success(f"Loaded saved results from: '{saved_dir_input}'")
    elif dataset_source == "Smart Importer (AI)":
        uploaded_file = st.file_uploader("Upload File for AI Parsing", type=["csv", "txt"])

    data_type = st.selectbox("Data Type", ['hyper_image', 'multi_spectra', 'single_spectrum'], index=0)
    
    use_glass = st.checkbox("Use Glass/Background Subtraction", value=True)
    glass_file_path = None
    if use_glass:
        glass_source = st.radio("Glass Background Source", ["Upload File", "Local File Path"], index=0)
        if glass_source == "Upload File":
            up_glass = st.file_uploader("Upload Glass File (.txt)", type=["txt"], key="glass_file_uploader")
            if up_glass is not None:
                glass_file_path = save_uploaded_file(up_glass, "temp_glass.txt")
        else:
            glass_file_path = st.text_input("Local Glass File Path (.txt)", value=st.session_state.get("glass_path_val", ""), key="glass_file_local")

# Container 2: Preset & Pipeline Method (expanded=True)
with st.sidebar.expander("⚡ Preset & Pipeline Method", expanded=True):
    preset = st.selectbox("Laser Preset", ["532 nm (Fingerprint + C-H stretch)", "785 nm (Fingerprint only)", "Custom"], index=0)
    
    if preset == "532 nm (Fingerprint + C-H stretch)":
        def_crop_low = 400.0
        def_crop_high = 3300.0
        def_skip_silent = True
        def_glass_method = "vector"
        def_airpls_strength = 1e3
        def_norm_mode = "dual"
    elif preset == "785 nm (Fingerprint only)":
        def_crop_low = 400.0
        def_crop_high = 1950.0
        def_skip_silent = False
        def_glass_method = "lsq"
        def_airpls_strength = 1e5
        def_norm_mode = "single"
    else: # Custom
        def_crop_low = 400.0
        def_crop_high = 3300.0
        def_skip_silent = True
        def_glass_method = "vector"
        def_airpls_strength = 1e3
        def_norm_mode = "dual"

    pipeline_analysis = st.selectbox("Analysis Method", ["VCA (Unmixing)", "PCA (Principal Components)", "HCA (Clustering)", "HDBSCAN"], index=0)

# Container 3: Preprocessing & Baseline Parameters (expanded=False)
with st.sidebar.expander("🛠️ Preprocessing & Baseline Parameters", expanded=False):
    crop_low = st.number_input("Crop Low (cm-1)", value=def_crop_low)
    crop_high = st.number_input("Crop High (cm-1)", value=def_crop_high)
    skip_silent = st.checkbox("Exclude Raman Silent Region (1900-2600 cm-1)", value=def_skip_silent)
    
    glass_methods = ["vector", "lsq", "direct", "None"]
    glass_method = st.selectbox("Glass Subtraction Method", glass_methods, index=glass_methods.index(def_glass_method) if def_glass_method in glass_methods else 0)
    
    cosmic_ray_threshold = st.slider("Cosmic Ray Threshold (z-score)", min_value=1.0, max_value=15.0, value=4.5, step=0.5)
    
    airpls_strength = st.number_input("airPLS Strength (lambda)", value=float(def_airpls_strength), format="%e")
    airpls_itermax = st.number_input("airPLS Max Iterations", min_value=10, max_value=200, value=50)
    
    norm_modes = ["dual", "single"]
    norm_mode = st.selectbox("Normalisation Mode", norm_modes, index=norm_modes.index(def_norm_mode) if def_norm_mode in norm_modes else 0)
    
    st.markdown("##### Spectral Smoothing")
    smooth_method = st.selectbox("Smoothing Method", ["None", "savgol", "gaussian"], index=0)
    smooth_savgol_window = 15
    smooth_savgol_polyorder = 3
    smooth_gaussian_sigma = 2.0
    if smooth_method == "savgol":
        smooth_savgol_window = st.number_input("Savgol Window Size (odd integer)", min_value=3, value=15, step=2)
        if smooth_savgol_window % 2 == 0:
            smooth_savgol_window += 1
        smooth_savgol_polyorder = st.number_input("Savgol Polynomial Order", min_value=1, max_value=smooth_savgol_window-1, value=3, step=1)
    elif smooth_method == "gaussian":
        smooth_gaussian_sigma = st.number_input("Gaussian Sigma (std dev)", min_value=0.1, value=2.0, step=0.1)

    st.markdown("##### Spatial Smoothing")
    smooth_spatial_sigma = st.number_input("Spatial Gaussian Sigma (0.0 to disable)", min_value=0.0, value=0.0, step=0.1)

# Container 4: Analysis Algorithm Parameters (expanded=False)
with st.sidebar.expander("📊 Analysis Algorithm Parameters", expanded=False):
    n_endmembers = 8
    endmember_labels_input = ""
    map_interpolation = "nearest"
    pca_components = 3
    hca_distance = "euclidean"
    hca_linkage = "ward"
    hca_dist = 1.0
    truncate_dendrogram = False
    truncate_p_val = None
    hdb_min_cluster_size = 5
    hdb_min_samples = 5
    
    if pipeline_analysis == "VCA (Unmixing)":
        n_endmembers = st.slider("VCA Endmembers", min_value=2, max_value=20, value=8)
        endmember_labels_input = st.text_input("Endmember Labels (comma-separated, optional)", value="")
        map_interpolation = st.selectbox("Abundance Map Interpolation", ["nearest", "bilinear", "none"], index=0)
    elif pipeline_analysis == "PCA (Principal Components)":
        pca_components = st.slider("PCA Components", min_value=1, max_value=20, value=3)
    elif pipeline_analysis == "HCA (Clustering)":
        hca_distance = st.selectbox("Distance Metric", ["euclidean", "cosine", "manhattan", "pearson"])
        if hca_distance in ["cosine", "manhattan"]:
            hca_linkage = st.selectbox("Linkage Method", ["complete", "average", "single"])
        else:
            hca_linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
        hca_dist = st.number_input("Distance Threshold (dist)", min_value=0.0, value=1.0, step=0.1)
        truncate_dendrogram = st.checkbox("Truncate Dendrogram View", value=False)
        if truncate_dendrogram:
            truncate_p_val = st.number_input("Number of Branches (p)", min_value=2, value=10, step=1)
    elif pipeline_analysis == "HDBSCAN":
        hdb_min_cluster_size = st.number_input("Min Cluster Size", value=5, min_value=2)
        hdb_min_samples = st.number_input("Min Samples", value=5, min_value=1)

# Container 4.5: Spatial Map Alignment & 2D Crop (expanded=False)
with st.sidebar.expander("🔄 Spatial Map Alignment & 2D Crop", expanded=False):
    map_rotation = st.selectbox("Rotate Map", [0, 90, 180, 270], index=0, format_func=lambda x: f"{x}°", help="Rotate spatial maps by 0, 90, 180, or 270 degrees.")
    map_flip_h = st.checkbox("Flip Horizontally (Left ↔ Right)", value=False, help="Mirror spatial maps left-to-right.")
    map_flip_v = st.checkbox("Flip Vertically (Top ↕ Bottom)", value=False, help="Mirror spatial maps top-to-bottom.")
    
    st.markdown("---")
    crop_spatial_active = st.checkbox("Enable 2D Spatial Map Crop", value=False, help="Trim empty or uninformative outer spatial pixels.")
    crop_x_min, crop_x_max = 0, 10000
    crop_y_min, crop_y_max = 0, 10000
    if crop_spatial_active:
        col_cx, col_cy = st.columns(2)
        crop_x_min = col_cx.number_input("X Min Pixel", min_value=0, value=0, step=1)
        crop_x_max = col_cx.number_input("X Max Pixel", min_value=1, value=1000, step=1)
        crop_y_min = col_cy.number_input("Y Min Pixel", min_value=0, value=0, step=1)
        crop_y_max = col_cy.number_input("Y Max Pixel", min_value=1, value=1000, step=1)

# Container 5: Export & Output Settings (expanded=False)
with st.sidebar.expander("💾 Export & Output Settings", expanded=False):
    custom_output_dir = st.text_input("Output Directory Path", value="./export_results")
    
    st.markdown("##### Optics / Acquisition Metadata (Optional)")
    laser_wavelength = st.text_input("Laser Wavelength", value="", help="e.g. 532 nm")
    integration_time = st.number_input("Integration Time (s)", min_value=0.0, value=0.0, step=0.1)
    laser_power = st.number_input("Laser Power (mW)", min_value=0.0, value=0.0, step=1.0)
    objective = st.text_input("Objective", value="", help="e.g. 100x / 0.9 NA")
    grating = st.text_input("Grating", value="", help="e.g. 600 g/mm")
    accumulations = st.number_input("Accumulations", min_value=0, value=0, step=1)
    
    manual_export_btn = st.button("💾 Save Copy to Output Directory", use_container_width=True)

workflow_steps = [
    "Step 1: 📂 Data Input & Selection",
    "Step 2: 🗺️ Spatial Mapping & Co-localization",
    "Step 3: 📈 Reference Spectra & Peak Analysis",
    "Step 4: 📊 Quantification & Downstream Statistics",
    "Step 5: 💾 Export & Data Table Inspection"
]

def update_workflow_step(new_idx: int):
    """Unified step update logic: updates current_step_index and synchronizes radio widget keys."""
    if 0 <= new_idx < len(workflow_steps):
        st.session_state["current_step_index"] = new_idx
        st.session_state["sidebar_step_nav_radio"] = workflow_steps[new_idx]
        st.session_state["top_step_nav_radio"] = workflow_steps[new_idx]

def on_sidebar_step_nav_change():
    selected = st.session_state.get("sidebar_step_nav_radio")
    if selected in workflow_steps:
        update_workflow_step(workflow_steps.index(selected))

def on_top_step_nav_change():
    selected = st.session_state.get("top_step_nav_radio")
    if selected in workflow_steps:
        update_workflow_step(workflow_steps.index(selected))

if "current_step_index" not in st.session_state:
    st.session_state["current_step_index"] = 0

init_idx = st.session_state["current_step_index"]
if "sidebar_step_nav_radio" not in st.session_state:
    st.session_state["sidebar_step_nav_radio"] = workflow_steps[init_idx]
if "top_step_nav_radio" not in st.session_state:
    st.session_state["top_step_nav_radio"] = workflow_steps[init_idx]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Workflow Stepper")
selected_sidebar_step = st.sidebar.radio(
    "Active Step",
    workflow_steps,
    index=st.session_state["current_step_index"],
    key="sidebar_step_nav_radio",
    on_change=on_sidebar_step_nav_change
)

# ==============================================================================
# DATA LOADING & PIPELINE EXECUTION ENGINE
# ==============================================================================

def load_dataset_matrix(dataset_source, selected_sample, uploaded_file, local_scan_path, data_type):
    """Load dataset into common format: (wavenumber, matrix, nrows, ncols, position, label, data_name, sp_obj)"""
    if dataset_source == "Sample Datasets":
        if not selected_sample or selected_sample == "-- Select --":
            return None
        file_path = os.path.join(os.path.dirname(__file__), 'data', selected_sample)
        if not os.path.exists(file_path):
            return None
            
        data_name = selected_sample.split('.')[0]
        if selected_sample.endswith('.csv.xz'):
            base_path = file_path[:-7]
            sp_obj = sp.hyper_object(data_name, data_type=data_type)
            try:
                sp_obj.read_csv_xz(base_path)
            except Exception as e:
                if 'z' in str(e):
                    sp_obj.read_csv_3d_xz(base_path)
                else:
                    raise e
        elif selected_sample.endswith('.spc'):
            base_path = file_path[:-4]
            sp_obj = sp.hyper_object(data_name, data_type=data_type)
            sp_obj.read_spc(base_path)
        elif selected_sample.endswith('.csv') or selected_sample.endswith('.txt'):
            df = pd.read_csv(file_path)
            sp_obj = sp.hyper_object(data_name, data_type=data_type)
            sp_obj.data = df.drop(columns=['label', 'x', 'y', 'z'], errors='ignore')
            if 'x' in df.columns and 'y' in df.columns:
                sp_obj.position = df[['x', 'y']]
            else:
                sp_obj.position = pd.DataFrame({'x': np.arange(len(df)), 'y': np.zeros(len(df))})
            sp_obj.m = int(pd.to_numeric(sp_obj.position['x']).max() + 1)
            sp_obj.n = int(pd.to_numeric(sp_obj.position['y']).max() + 1)
            sp_obj.label = pd.Series(df['label']) if 'label' in df.columns else pd.Series([1]*len(df))
        else:
            return None
            
        wavenumber = pd.to_numeric(sp_obj.data.columns).values
        matrix = sp_obj.data.values.T
        ncols = getattr(sp_obj, 'm', int(np.sqrt(matrix.shape[1])))
        nrows = getattr(sp_obj, 'n', int(np.sqrt(matrix.shape[1])))
        if ncols * nrows != matrix.shape[1]:
            nrows = matrix.shape[1] // ncols if ncols > 0 else 1
            
        return wavenumber, matrix, nrows, ncols, sp_obj.position, getattr(sp_obj, 'label', None), data_name, sp_obj

    elif dataset_source == "Upload Custom File":
        if uploaded_file is None:
            return None
        temp_path = save_uploaded_file(uploaded_file, uploaded_file.name)
        data_name = Path(uploaded_file.name).stem
        if uploaded_file.name.endswith('.txt'):
            try:
                wavenumber, matrix, ncols, nrows = wrp.load_witec_map(temp_path)
                xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
                position = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten()})
                label = pd.Series([1]*matrix.shape[1])
                sp_obj = sp.hyper_object(data_name, data_type=data_type)
                sp_obj.data = pd.DataFrame(matrix.T, columns=wavenumber)
                sp_obj.position = position
                sp_obj.m = ncols
                sp_obj.n = nrows
                return wavenumber, matrix, nrows, ncols, position, label, data_name, sp_obj
            except Exception:
                pass
                
        sp_obj = sp.hyper_object(data_name, data_type=data_type)
        if uploaded_file.name.endswith('.csv.xz'):
            base_path = temp_path[:-7]
            try:
                sp_obj.read_csv_xz(base_path)
            except Exception:
                sp_obj.read_csv_3d_xz(base_path)
        elif uploaded_file.name.endswith('.spc'):
            base_path = temp_path[:-4]
            sp_obj.read_spc(base_path)
        else:
            df = pd.read_csv(temp_path)
            sp_obj.data = df.drop(columns=['label', 'x', 'y', 'z'], errors='ignore')
            if 'x' in df.columns and 'y' in df.columns:
                sp_obj.position = df[['x', 'y']]
            else:
                sp_obj.position = pd.DataFrame({'x': np.arange(len(df)), 'y': np.zeros(len(df))})
            sp_obj.m = int(pd.to_numeric(sp_obj.position['x']).max() + 1)
            sp_obj.n = int(pd.to_numeric(sp_obj.position['y']).max() + 1)
            sp_obj.label = pd.Series(df['label']) if 'label' in df.columns else pd.Series([1]*len(df))
            
        wavenumber = pd.to_numeric(sp_obj.data.columns).values
        matrix = sp_obj.data.values.T
        ncols = getattr(sp_obj, 'm', int(np.sqrt(matrix.shape[1])))
        nrows = getattr(sp_obj, 'n', int(np.sqrt(matrix.shape[1])))
        return wavenumber, matrix, nrows, ncols, sp_obj.position, getattr(sp_obj, 'label', None), data_name, sp_obj

    elif dataset_source == "Local Scan File (.txt)":
        if not local_scan_path or not os.path.exists(local_scan_path):
            return None
        data_name = Path(local_scan_path).stem
        wavenumber, matrix, ncols, nrows = wrp.load_witec_map(local_scan_path)
        xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
        position = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten()})
        label = pd.Series([1]*matrix.shape[1])
        sp_obj = sp.hyper_object(data_name, data_type=data_type)
        sp_obj.data = pd.DataFrame(matrix.T, columns=wavenumber)
        sp_obj.position = position
        sp_obj.m = ncols
        sp_obj.n = nrows
        return wavenumber, matrix, nrows, ncols, position, label, data_name, sp_obj

    elif dataset_source == "Smart Importer (AI)":
        if uploaded_file is None:
            return None
        if parse_with_ollama is None:
            st.error("Smart Importer (AI) module not available.")
            return None
        try:
            df, code = parse_with_ollama(uploaded_file.getvalue(), uploaded_file.name)
        except Exception as e:
            st.warning(f"Ollama AI service is offline or returned an error: {e}. Falling back to standard CSV parser.")
            try:
                df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
            except Exception as parse_err:
                st.error(f"Fallback CSV parsing failed: {parse_err}")
                return None
        if df is None or df.empty:
            st.error("Parsed dataset is empty.")
            return None
        data_name = Path(uploaded_file.name).stem
        sp_obj = sp.hyper_object(data_name, data_type=data_type)
        spectral_df = df.drop(columns=['label', 'x', 'y', 'z'], errors='ignore')
        
        num_cols = [c for c in spectral_df.columns if str(c).replace('.','',1).replace('-','',1).isdigit()]
        if not num_cols:
            num_cols = list(spectral_df.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            st.error("No numeric spectral wavenumber columns found in parsed data.")
            return None
            
        sp_obj.data = spectral_df[num_cols].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if 'x' in df.columns and 'y' in df.columns:
            sp_obj.position = df[['x', 'y']].apply(pd.to_numeric, errors='coerce').fillna(0)
        else:
            sp_obj.position = pd.DataFrame({'x': np.arange(len(df)), 'y': np.zeros(len(df))})
            
        max_x = pd.to_numeric(sp_obj.position['x'], errors='coerce').max()
        max_y = pd.to_numeric(sp_obj.position['y'], errors='coerce').max()
        sp_obj.m = int(max(1, max_x + 1)) if not np.isnan(max_x) else 1
        sp_obj.n = int(max(1, max_y + 1)) if not np.isnan(max_y) else 1
        sp_obj.label = pd.Series(df['label']) if 'label' in df.columns else pd.Series([1]*len(df))
        
        wavenumber = pd.to_numeric(sp_obj.data.columns, errors='coerce').values
        matrix = sp_obj.data.values.T
        return wavenumber, matrix, sp_obj.n, sp_obj.m, sp_obj.position, sp_obj.label, data_name, sp_obj

def save_fig_multiformat(fig, path, dpi=150):
    base = Path(path).with_suffix("")
    for fmt in [".png", ".pdf", ".svg"]:
        fig.savefig(base.with_suffix(fmt), dpi=dpi, bbox_inches="tight", pad_inches=0.1)


def run_pipeline_core(wavenumber, matrix, nrows, ncols, position, label, data_name, sp_obj,
                      crop_low, crop_high, skip_silent, glass_method, use_glass, glass_file_path,
                      cosmic_ray_threshold, airpls_strength, airpls_itermax, norm_mode,
                      smooth_method, smooth_savgol_window, smooth_savgol_polyorder, smooth_gaussian_sigma, smooth_spatial_sigma,
                      pipeline_analysis, n_endmembers, endmember_labels_input, map_interpolation,
                      pca_components, hca_distance, hca_linkage, hca_dist, truncate_dendrogram, truncate_p_val,
                      hdb_min_cluster_size, hdb_min_samples,
                      custom_output_dir, laser_wavelength, integration_time, laser_power, objective, grating, accumulations):
    """Core processing function executing steps 1 through 10."""
    raw_wavenumber = wavenumber.copy()
    raw_matrix = matrix.copy()
    
    # 0. Crop bounds validation
    if crop_low >= crop_high:
        st.error(f"Invalid Crop Bounds: Crop Low ({crop_low} cm⁻¹) must be strictly less than Crop High ({crop_high} cm⁻¹).")
        raise ValueError(f"Crop low ({crop_low}) >= crop high ({crop_high})")

    # 1. Cosmic ray removal
    if cosmic_ray_threshold > 0:
        matrix, n_fixed = wrp.remove_cosmic_rays(matrix, nrows, ncols, threshold=cosmic_ray_threshold)
        
    # 2. Glass background subtraction
    run_use_glass = use_glass and glass_file_path and os.path.exists(glass_file_path) and glass_method != "None"
    glass_wn, glass_int = None, None
    if run_use_glass:
        try:
            glass_wn, glass_int = wrp.load_spectrum(glass_file_path)
            glass_interp = np.interp(wavenumber, glass_wn, glass_int)
            if glass_method == "direct":
                matrix = wrp.subtract_glass_direct(matrix, glass_interp)
            elif glass_method == "vector":
                matrix = wrp.subtract_glass_vector(matrix, glass_interp)
            elif glass_method == "lsq":
                matrix = wrp.subtract_glass_lsq(matrix, glass_interp)
        except Exception as ge:
            st.warning(f"Glass reference is flat or singular; skipping glass subtraction. ({ge})")
            run_use_glass = False
            
    # 3. Spatial Gaussian smoothing
    if smooth_spatial_sigma > 0.0:
        matrix = wrp.spatial_gaussian_smooth(matrix, nrows, ncols, smooth_spatial_sigma)
        
    # 4. Spectral smoothing
    if smooth_method and smooth_method != "None":
        matrix = wrp.smooth_spectra(matrix, smooth_method, window=smooth_savgol_window, polyorder=smooth_savgol_polyorder, sigma=smooth_gaussian_sigma)
        
    # 5. Crop spectrum
    try:
        wavenumber, matrix = wrp.crop_spectrum(wavenumber, matrix, low=crop_low, high=crop_high, skip_silent=skip_silent)
    except ValueError as ve:
        st.error(f"Spectral cropping failed: {ve}")
        raise ve
        
    if matrix.size == 0 or wavenumber.size == 0:
        st.error("Matrix or wavenumber slice is empty after preprocessing.")
        raise ValueError("Empty data matrix after crop")

    # 6. Baseline correction (airPLS)
    matrix = wrp.correct_baseline(matrix, lam=airpls_strength, itermax=airpls_itermax)
    
    # 7. Normalisation
    matrix = wrp.normalise(matrix, wavenumber, mode=norm_mode)

    # 8. Rank validation for downstream analysis methods
    n_channels, n_pixels = matrix.shape
    max_rank = min(n_channels, n_pixels)
    if pipeline_analysis == "PCA (Principal Components)":
        if pca_components > max_rank:
            st.warning(f"Requested PCA components ({pca_components}) exceeds dataset rank limit ({max_rank}). Automatically adjusting to {max_rank}.")
            pca_components = max_rank
    elif pipeline_analysis == "VCA (Unmixing)":
        if n_endmembers > max_rank:
            st.warning(f"Requested VCA endmembers ({n_endmembers}) exceeds dataset rank limit ({max_rank}). Automatically adjusting to {max_rank}.")
            n_endmembers = max_rank
    elif pipeline_analysis == "HDBSCAN":
        if hdb_min_cluster_size > n_pixels:
            st.warning(f"Min Cluster Size ({hdb_min_cluster_size}) exceeds total pixels ({n_pixels}). Automatically adjusting to {n_pixels}.")
            hdb_min_cluster_size = n_pixels
    
    # Output paths setup
    out_root = Path(custom_output_dir if custom_output_dir else "./export_results")
    fig_dir = out_root / "figures"
    proc_dir = out_root / "processed"
    fig_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_meta = {
        "SCAN_FILE": data_name,
        "GLASS_FILE": glass_file_path if run_use_glass else None,
        "LASER_WAVELENGTH": laser_wavelength if laser_wavelength else None,
        "INTEGRATION_TIME_SEC": integration_time if integration_time > 0 else None,
        "LASER_POWER_MW": laser_power if laser_power > 0 else None,
        "OBJECTIVE": objective if objective else None,
        "GRATING": grating if grating else None,
        "ACCUMULATIONS": accumulations if accumulations > 0 else None,
        "CROP_LOW": crop_low,
        "CROP_HIGH": crop_high,
        "SKIP_SILENT": skip_silent,
        "GLASS_METHOD": glass_method if run_use_glass else None,
        "COSMIC_RAY_THRESHOLD": cosmic_ray_threshold,
        "AIRPLS_STRENGTH": airpls_strength,
        "AIRPLS_ITERMAX": airpls_itermax,
        "NORM_MODE": norm_mode,
        "SMOOTH_METHOD": smooth_method if smooth_method != "None" else None,
        "SMOOTH_SAVGOL_WINDOW": smooth_savgol_window if smooth_method == "savgol" else None,
        "SMOOTH_SAVGOL_POLYORDER": smooth_savgol_polyorder if smooth_method == "savgol" else None,
        "SMOOTH_GAUSSIAN_SIGMA": smooth_gaussian_sigma if smooth_method == "gaussian" else None,
        "SPATIAL_GAUSSIAN_SIGMA": smooth_spatial_sigma,
        "PCA_COMPONENTS": pca_components if pipeline_analysis == "PCA (Principal Components)" else None,
        "N_ENDMEMBERS": n_endmembers if pipeline_analysis == "VCA (Unmixing)" else None,
        "MAP_INTERPOLATION": map_interpolation if pipeline_analysis == "VCA (Unmixing)" else None,
        "HCA_DISTANCE": hca_distance if pipeline_analysis == "HCA (Clustering)" else None,
        "HCA_LINKAGE": hca_linkage if pipeline_analysis == "HCA (Clustering)" else None,
        "HCA_DIST": hca_dist if pipeline_analysis == "HCA (Clustering)" else None,
        "HCA_TRUNCATE_P": truncate_p_val if pipeline_analysis == "HCA (Clustering)" else None
    }
    meta, header_str = wrp.get_metadata(cfg_meta, ncols, nrows)
    
    # Preprocessed spectra export
    df_preprocessed = pd.DataFrame(matrix.T, columns=wavenumber)
    with open(proc_dir / "preprocessed_spectra.csv", "w") as fh:
        fh.write(header_str)
        df_preprocessed.to_csv(fh, index=False)
    
    res = {
        "data_name": data_name,
        "wavenumber": wavenumber,
        "matrix": matrix,
        "raw_wavenumber": raw_wavenumber,
        "raw_matrix": raw_matrix,
        "nrows": nrows,
        "ncols": ncols,
        "position": position,
        "label": label,
        "metadata": meta,
        "out_root": str(out_root),
        "run_use_glass": run_use_glass,
        "glass_wn": glass_wn,
        "glass_int": glass_int,
        "skip_silent": skip_silent,
        "analysis_method": pipeline_analysis
    }
    
    # 8. Run Analysis Algorithm & Export Artifacts
    if pipeline_analysis == "VCA (Unmixing)":
        Ae = wrp.vca(matrix, n_endmembers)
        abundances = wrp.compute_abundances(matrix, Ae, nrows, ncols)
        
        endmember_path = str(proc_dir / "endmember_spectra.csv")
        abundance_path = str(proc_dir / "abundance_maps.csv")
        parsed_labels = [lbl.strip() for lbl in endmember_labels_input.split(",")] if endmember_labels_input else None
        
        wrp.export_endmembers(Ae, wavenumber, endmember_path, labels=parsed_labels, header_str=header_str)
        wrp.export_abundances(abundances, nrows, ncols, abundance_path, labels=parsed_labels, header_str=header_str)
        
        if run_use_glass and glass_wn is not None:
            wrp.plot_glass(glass_wn, glass_int, str(fig_dir / "glass_spectrum.png"), 150, skip_silent)
        wrp.plot_endmembers(Ae, wavenumber, skip_silent, str(fig_dir / "vca_endmembers.png"), 150, labels=parsed_labels)
        wrp.plot_abundance_maps(abundances, str(fig_dir / "abundance_maps.png"), 150, labels=parsed_labels, interpolation=map_interpolation, rotation=map_rotation, flip_h=map_flip_h, flip_v=map_flip_v, crop_spatial=crop_spatial_active, crop_x_min=crop_x_min, crop_x_max=crop_x_max, crop_y_min=crop_y_min, crop_y_max=crop_y_max)
        
        # Pearson Correlation Matrix & Heatmap Export
        em_names = [parsed_labels[i] if (parsed_labels and i < len(parsed_labels)) else f"Endmember {i+1}" for i in range(n_endmembers)]
        corr_matrix = np.corrcoef(Ae.T)
        df_corr = pd.DataFrame(corr_matrix, index=em_names, columns=em_names)
        corr_path = proc_dir / "correlation_matrix.csv"
        with open(corr_path, "w") as fh:
            fh.write(header_str)
            df_corr.to_csv(fh)
            
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        im_corr = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax_corr.set_xticks(np.arange(len(em_names)))
        ax_corr.set_yticks(np.arange(len(em_names)))
        ax_corr.set_xticklabels(em_names, rotation=45, ha="right", fontsize=8)
        ax_corr.set_yticklabels(em_names, fontsize=8)
        for i_idx in range(len(em_names)):
            for j_idx in range(len(em_names)):
                ax_corr.text(j_idx, i_idx, f"{corr_matrix[i_idx, j_idx]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold",
                             color="white" if abs(corr_matrix[i_idx, j_idx]) > 0.4 else "black")
        ax_corr.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold")
        fig_corr.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04)
        save_fig_multiformat(fig_corr, fig_dir / "correlation_heatmap.png", 150)
        plt.close(fig_corr)

        # Biochemical Ratios CSV & Figure Export
        ratio_definitions = [
            ("Lipid_Protein_2850_2930", 2850.0, 2930.0),
            ("LipidEster_ProteinAmideI_1740_1660", 1740.0, 1660.0),
            ("Lipid_ProteinFingerprint_1440_1660", 1440.0, 1660.0),
            ("ProteinPurity_1003_1660", 1003.0, 1660.0),
            ("DNA_Protein_785_1003", 785.0, 1003.0)
        ]
        ratio_data = {}
        for r_name, w1, w2 in ratio_definitions:
            idx1 = np.abs(wavenumber - w1).argmin()
            idx2 = np.abs(wavenumber - w2).argmin()
            ratios = Ae[idx1, :] / np.where(Ae[idx2, :] == 0, 1e-10, Ae[idx2, :])
            ratio_data[r_name] = ratios
        df_ratios = pd.DataFrame(ratio_data, index=em_names)
        ratios_path = proc_dir / "biochemical_ratios.csv"
        with open(ratios_path, "w") as fh:
            fh.write(header_str)
            df_ratios.to_csv(fh)

        fig_bio, ax_bio = plt.subplots(figsize=(8, 5))
        df_ratios.plot(kind="bar", ax=ax_bio, colormap="tab10", width=0.8)
        ax_bio.set_ylabel("Peak Intensity Ratio", fontsize=10)
        ax_bio.set_title("Biochemical Macromolecular Peak Ratios", fontsize=12, fontweight="bold")
        ax_bio.set_xticklabels(em_names, rotation=45, ha="right", fontsize=9)
        ax_bio.grid(axis="y", ls="--", alpha=0.3)
        ax_bio.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
        save_fig_multiformat(fig_bio, fig_dir / "biochemical_ratios.png", 150)
        plt.close(fig_bio)

        df_em = pd.read_csv(endmember_path, comment="#")
        df_ab = pd.read_csv(abundance_path, comment="#")
        
        res.update({
            "Ae": Ae,
            "abundances": abundances,
            "n_endmembers": n_endmembers,
            "parsed_labels": parsed_labels,
            "map_interpolation": map_interpolation,
            "df_endmembers": df_em,
            "df_abundances": df_ab
        })

    elif pipeline_analysis == "PCA (Principal Components)":
        scores, loadings, variance_ratio = wrp.run_pca(matrix, pca_components)
        
        xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
        pos_df = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten()})
        scores_df = pd.concat([pos_df, pd.DataFrame(scores.T, columns=[f"PC {i+1}" for i in range(pca_components)])], axis=1)
        loadings_df = pd.DataFrame(loadings, index=[f"PC {i+1}" for i in range(pca_components)], columns=wavenumber)
        
        scores_path = proc_dir / "pca_scores.csv"
        loadings_path = proc_dir / "pca_loadings.csv"
        
        with open(scores_path, "w") as fh:
            fh.write(header_str)
            scores_df.to_csv(fh, index=False)
        with open(loadings_path, "w") as fh:
            fh.write(header_str)
            loadings_df.to_csv(fh, index=True)

        # Export PCA Stacked Loadings Plot (with left margin=0.12 and no clipping)
        fig_stack, axs_s = plt.subplots(pca_components, sharex='all', sharey='all', figsize=(10, 1.8 * pca_components),
                                        gridspec_kw={'hspace': 0, 'left': 0.12, 'bottom': 0.15, 'right': 0.95, 'top': 0.9})
        axs_s = np.atleast_1d(axs_s).flatten()
        wn_num = pd.to_numeric(wavenumber)
        cmap_pca = plt.cm.tab10(np.linspace(0, 1, max(10, pca_components)))
        for i in range(pca_components):
            color = cmap_pca[i % 10]
            axs_s[i].plot(wn_num, loadings[i, :], color=color, lw=1.5)
            axs_s[i].set_ylabel(f"PC {i+1}", fontsize=9, fontweight="bold")
            axs_s[i].grid(ls="--", alpha=0.3)
            axs_s[i].axhline(0, color="gray", ls="--", alpha=0.5)
        axs_s[-1].set_xlabel("Wavenumber (cm-1)", fontsize=10)
        axs_s[0].set_title("PCA Loadings Stacked Plot", fontsize=12, fontweight="bold")
        save_fig_multiformat(fig_stack, fig_dir / "pca_loadings_stacked.png", 150)
        plt.close(fig_stack)

        # Export PCA Score Maps Grid
        cols_per_row = min(4, pca_components)
        nrows_fig = (pca_components + cols_per_row - 1) // cols_per_row
        fig_grid, axes_g = plt.subplots(nrows_fig, cols_per_row, figsize=(cols_per_row * 4, nrows_fig * 3.5))
        axes_g = np.atleast_1d(axes_g).flatten()
        for i in range(pca_components):
            score_map = crop_and_orient_map(scores[i, :].reshape(nrows, ncols), rotation=map_rotation, flip_h=map_flip_h, flip_v=map_flip_v, crop_spatial=crop_spatial_active, crop_x_min=crop_x_min, crop_x_max=crop_x_max, crop_y_min=crop_y_min, crop_y_max=crop_y_max)
            im_g = axes_g[i].imshow(score_map, cmap="viridis", interpolation="nearest")
            axes_g[i].set_title(f"PC {i+1} ({variance_ratio[i]*100:.1f}%)", fontsize=10, fontweight="bold")
            axes_g[i].axis("off")
            fig_grid.colorbar(im_g, ax=axes_g[i], fraction=0.046, pad=0.04)
        for ax in axes_g[pca_components:]:
            ax.axis("off")
        save_fig_multiformat(fig_grid, fig_dir / "pca_score_maps_grid.png", 150)
        plt.close(fig_grid)

        # Individual PCA Score Maps & Loadings
        for i in range(pca_components):
            fig_single, ax_single = plt.subplots(figsize=(5, 4))
            score_map_single = crop_and_orient_map(scores[i, :].reshape(nrows, ncols), rotation=map_rotation, flip_h=map_flip_h, flip_v=map_flip_v, crop_spatial=crop_spatial_active, crop_x_min=crop_x_min, crop_x_max=crop_x_max, crop_y_min=crop_y_min, crop_y_max=crop_y_max)
            im_s = ax_single.imshow(score_map_single, cmap="viridis", interpolation="nearest")
            ax_single.set_title(f"PC {i+1} Score Map", fontsize=10, fontweight="bold")
            ax_single.axis("off")
            fig_single.colorbar(im_s, ax=ax_single, fraction=0.046, pad=0.04)
            save_fig_multiformat(fig_single, fig_dir / f"pca_score_map_pc{i+1}.png", 150)
            plt.close(fig_single)
            
            fig_l, ax_l = plt.subplots(figsize=(6, 3.5))
            ax_l.plot(wn_num, loadings[i, :], lw=1.5)
            ax_l.set_xlabel("Wavenumber (cm-1)")
            ax_l.set_ylabel("Loading")
            ax_l.set_title(f"PC {i+1} Loading Spectrum", fontsize=10, fontweight="bold")
            ax_l.grid(ls="--", alpha=0.3)
            save_fig_multiformat(fig_l, fig_dir / f"pca_loadings_pc{i+1}.png", 150)
            plt.close(fig_l)

        # PCA Scatter Plot (PC1 vs PC2)
        if pca_components >= 2:
            fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
            ax_sc.scatter(scores[0, :], scores[1, :], alpha=0.7, color="#1f77b4", edgecolors="none")
            ax_sc.set_xlabel("PC 1")
            ax_sc.set_ylabel("PC 2")
            ax_sc.set_title("PCA Score Scatter (PC 1 vs PC 2)", fontsize=12, fontweight="bold")
            ax_sc.grid(ls="--", alpha=0.3)
            save_fig_multiformat(fig_sc, fig_dir / "pca_scatter.png", 150)
            plt.close(fig_sc)
            
        res.update({
            "pca_scores": scores,
            "pca_loadings": loadings,
            "pca_variance": variance_ratio,
            "pca_components": pca_components,
            "df_pca_scores": scores_df,
            "df_pca_loadings": loadings_df
        })

    elif pipeline_analysis == "HCA (Clustering)":
        hca_obj = sp.hyper_object("hca_analysis")
        hca_obj.data = pd.DataFrame(matrix.T, columns=wavenumber)
        xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
        hca_obj.position = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten()})
        hca_obj.m = ncols
        hca_obj.n = nrows
        hca_obj.resolution = 1
        hca_obj.sublabel = pd.Series(np.zeros(matrix.shape[1]), name="sublabel")
        hca_obj.label = pd.Series([1]*matrix.shape[1])
        
        plt.close('all')
        hca_obj.hca(hca_distance, hca_linkage, hca_dist, truncate_p_val)
        hca_fig = plt.gcf()
        
        export_df = pd.concat([hca_obj.position, hca_obj.label, hca_obj.data], axis=1)
        hca_csv_path = str(proc_dir / "hca_clustering.csv")
        with open(hca_csv_path, "w") as fh:
            fh.write(header_str)
            export_df.to_csv(fh, index=False)

        if hca_fig:
            save_fig_multiformat(hca_fig, fig_dir / "hca_dendrogram.png", 150)
            
        try:
            plt.figure(figsize=(6, 5))
            colors = hca_obj.show_map('auto', None, 1, rotation=map_rotation, flip_h=map_flip_h, flip_v=map_flip_v)
            fig_map = plt.gcf()
            save_fig_multiformat(fig_map, fig_dir / "hca_cluster_map.png", 150)
            plt.close(fig_map)
        except Exception:
            pass

        try:
            wn_hca = pd.to_numeric(hca_obj.data.columns)
            unique_c = sorted(hca_obj.label.unique())
            fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
            cmap_h = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_c))))
            for c_idx, c_val in enumerate(unique_c):
                mask_c = hca_obj.label == c_val
                c_data = hca_obj.data[mask_c]
                if len(c_data) > 0:
                    m_spec = c_data.mean(axis=0).values
                    s_spec = c_data.std(axis=0).values
                    color_c = cmap_h[c_idx % 10]
                    ax_avg.plot(wn_hca, m_spec, label=f"Cluster {c_val} (n={len(c_data)})", color=color_c, lw=1.8)
                    ax_avg.fill_between(wn_hca, m_spec - s_spec, m_spec + s_spec, color=color_c, alpha=0.15)
            ax_avg.set_xlabel("Wavenumber (cm-1)")
            ax_avg.set_ylabel("Intensity (a.u.)")
            ax_avg.set_title("Cluster Mean Spectra (Mean ± Standard Deviation)", fontsize=12, fontweight="bold")
            ax_avg.legend(frameon=True)
            ax_avg.grid(ls="--", alpha=0.3)
            save_fig_multiformat(fig_avg, fig_dir / "hca_cluster_average_spectra.png", 150)
            plt.close(fig_avg)
        except Exception:
            pass
            
        res.update({
            "hca_obj": hca_obj,
            "hca_fig": hca_fig,
            "df_hca": pd.read_csv(hca_csv_path, comment="#")
        })

    elif pipeline_analysis == "HDBSCAN":
        hdb_obj = sp.hyper_object("hdbscan_analysis")
        hdb_obj.data = pd.DataFrame(matrix.T, columns=wavenumber)
        xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
        hdb_obj.position = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten()})
        hdb_obj.m = ncols
        hdb_obj.n = nrows
        hdb_obj.resolution = 1
        hdb_obj.sublabel = pd.Series(np.zeros(matrix.shape[1]), name="sublabel")
        hdb_obj.label = pd.Series([1]*matrix.shape[1])
        
        hdb_obj.hdbscan(hdb_min_cluster_size, hdb_min_samples)
        
        export_df = pd.concat([hdb_obj.position, hdb_obj.label, hdb_obj.data], axis=1)
        hdb_csv_path = str(proc_dir / "hdbscan_clustering.csv")
        with open(hdb_csv_path, "w") as fh:
            fh.write(header_str)
            export_df.to_csv(fh, index=False)

        try:
            plt.figure(figsize=(6, 5))
            colors = hdb_obj.show_map('auto', None, 1, rotation=map_rotation, flip_h=map_flip_h, flip_v=map_flip_v)
            fig_map = plt.gcf()
            save_fig_multiformat(fig_map, fig_dir / "hdbscan_cluster_map.png", 150)
            plt.close(fig_map)
        except Exception:
            pass
            
        res.update({
            "hdb_obj": hdb_obj,
            "df_hdbscan": pd.read_csv(hdb_csv_path, comment="#")
        })
        
    with open(proc_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=4)
        
    return res

state_key = (
    dataset_source,
    selected_sample,
    uploaded_file.name if (uploaded_file and hasattr(uploaded_file, 'name')) else None,
    local_scan_path,
    data_type,
    use_glass,
    glass_file_path,
    preset,
    pipeline_analysis,
    crop_low,
    crop_high,
    skip_silent,
    glass_method,
    cosmic_ray_threshold,
    airpls_strength,
    airpls_itermax,
    norm_mode,
    smooth_method,
    smooth_savgol_window,
    smooth_savgol_polyorder,
    smooth_gaussian_sigma,
    smooth_spatial_sigma,
    n_endmembers,
    endmember_labels_input,
    map_interpolation,
    pca_components,
    hca_distance,
    hca_linkage,
    hca_dist,
    truncate_dendrogram,
    truncate_p_val,
    hdb_min_cluster_size,
    hdb_min_samples,
    custom_output_dir
)

# Reactive execution trigger
if dataset_source != "📂 Saved Results Directory":
    if st.session_state.get("last_state_key") != state_key:
        dataset_tuple = load_dataset_matrix(dataset_source, selected_sample, uploaded_file, local_scan_path, data_type)
        if dataset_tuple is not None:
            wavenumber, matrix, nrows, ncols, position, label, data_name, sp_obj = dataset_tuple
            with st.spinner(f"⚡ Auto-executing pipeline on dataset '{data_name}'..."):
                try:
                    res = run_pipeline_core(
                        wavenumber, matrix, nrows, ncols, position, label, data_name, sp_obj,
                        crop_low, crop_high, skip_silent, glass_method, use_glass, glass_file_path,
                        cosmic_ray_threshold, airpls_strength, airpls_itermax, norm_mode,
                        smooth_method, smooth_savgol_window, smooth_savgol_polyorder, smooth_gaussian_sigma, smooth_spatial_sigma,
                        pipeline_analysis, n_endmembers, endmember_labels_input, map_interpolation,
                        pca_components, hca_distance, hca_linkage, hca_dist, truncate_dendrogram, truncate_p_val,
                        hdb_min_cluster_size, hdb_min_samples,
                        custom_output_dir, laser_wavelength, integration_time, laser_power, objective, grating, accumulations
                    )
                    st.session_state.pipeline_results = res
                    st.session_state.pipeline_success = True
                    st.session_state.last_state_key = state_key
                except Exception as e:
                    st.error(f"Pipeline execution failed: {e}")
                    st.session_state.pipeline_success = False
        else:
            st.session_state.pipeline_success = False

# Manual copy export button trigger
if manual_export_btn and st.session_state.get("pipeline_success", False):
    st.sidebar.success(f"Results successfully saved to `{custom_output_dir}`")

# ==============================================================================
# SEQUENTIAL 5-STEP GUIDED ANALYTICAL WORKFLOW RENDERING
# ==============================================================================

# Top Horizontal Stepper Navigation Bar
top_step_choice = st.radio(
    "Analytical Workflow Stepper",
    workflow_steps,
    index=st.session_state.get("current_step_index", 0),
    horizontal=True,
    key="top_step_nav_radio",
    on_change=on_top_step_nav_change
)
current_step = st.session_state.get("current_step_index", 0)

if st.session_state.get("pipeline_success", False):
    res = st.session_state.pipeline_results
    analysis_method = res.get("analysis_method")
    data_name = res.get("data_name")
    
    # State synchronization initialization
    current_dataset = data_name
    current_n_em = res.get("n_endmembers")
    current_pca_comp = res.get("pca_components")

    dataset_changed = (
        st.session_state.get("last_synced_dataset") != current_dataset or
        st.session_state.get("last_synced_n_em") != current_n_em or
        st.session_state.get("last_synced_pca_comp") != current_pca_comp
    )

    if dataset_changed or "global_selected_endmembers" not in st.session_state:
        if analysis_method == "VCA (Unmixing)":
            labels_all = res.get("parsed_labels", [])
            n_em_all = res.get("n_endmembers", 8)
            em_options_all = [labels_all[i] if (labels_all and i < len(labels_all)) else f"Endmember {i+1}" for i in range(n_em_all)]
            st.session_state["global_selected_endmembers"] = em_options_all
            for k in ["step2_vca_select", "step3_vca_grid_select", "step3_vca_overlap_select", "tab1_vca_select", "tab2_vca_select_grid", "tab2_vca_select_overlap"]:
                st.session_state[k] = em_options_all

        if "pca_components" in res or analysis_method == "PCA (Principal Components)":
            pca_comp = res.get("pca_components", 3)
            pc_options_all = [f"PC {i+1}" for i in range(pca_comp)]
            st.session_state["global_selected_pcs"] = pc_options_all
            for k in ["step3_pca_select", "step4_pca_select", "tab1_pca_select", "tab2_pca_select"]:
                st.session_state[k] = pc_options_all

        st.session_state["last_synced_dataset"] = current_dataset
        st.session_state["last_synced_n_em"] = current_n_em
        st.session_state["last_synced_pca_comp"] = current_pca_comp

    st.markdown("---")

    # --------------------------------------------------------------------------
    # STEP 1: 📂 Data Input & Selection
    # --------------------------------------------------------------------------
    if current_step == 0:
        st.subheader("Step 1: 📂 Data Input & Selection")
        st.success(f"✅ Pipeline Execution Active: **{data_name}** | Method: **{analysis_method}** | Preset: **{preset}**")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Spatial Map Grid Size", f"{res.get('nrows', 0)} × {res.get('ncols', 0)}")
        wn_arr = res.get('wavenumber', [])
        col_info2.metric("Spectral Channels", len(wn_arr), f"{min(wn_arr):.0f} – {max(wn_arr):.0f} cm⁻¹" if len(wn_arr) > 0 else "")
        col_info3.metric("Endmembers / Components", res.get('n_endmembers') or res.get('pca_components') or 'N/A')
        
        st.markdown("---")
        col_btn1, col_btn2 = st.columns([3, 1])
        col_btn1.info("💡 Dataset loaded and preprocessed automatically. Use the sidebar controls to tweak preprocessing parameters or switch laser presets, or click below to inspect spatial maps.")
        if col_btn2.button("Proceed to Step 2: Spatial Mapping 🗺️ ➡️", key="btn_step1_next", type="primary", use_container_width=True):
            update_workflow_step(1)
            st.rerun()

    # --------------------------------------------------------------------------
    # STEP 2: 🗺️ Spatial Mapping & Co-localization
    # --------------------------------------------------------------------------
    elif current_step == 1:
        st.subheader("Step 2: 🗺️ Spatial Mapping & Co-localization")
        st.markdown("##### 🔄 Spatial Alignment, Orientation & 2D Crop Controls")
        col_or1, col_or2, col_or3, col_or4, col_or5 = st.columns(5)
        rot_val = col_or1.selectbox("Map Rotation", [0, 90, 180, 270], index=[0, 90, 180, 270].index(map_rotation), format_func=lambda x: f"{x}°", key="quick_rot")
        fliph_val = col_or2.checkbox("Flip Horizontally (Left ↔ Right)", value=map_flip_h, key="quick_fliph")
        flipv_val = col_or3.checkbox("Flip Vertically (Top ↕ Bottom)", value=map_flip_v, key="quick_flipv")
        map_cmap = col_or4.selectbox("Map Colormap", ["inferno", "viridis", "plasma", "magma", "cividis", "turbo", "coolwarm", "jet", "rainbow", "gray"], index=0, key="quick_cmap")
        crop_active_val = col_or5.checkbox("Crop Spatial Map", value=crop_spatial_active, key="quick_crop_active")
        
        c_xmin, c_xmax, c_ymin, c_ymax = crop_x_min, crop_x_max, crop_y_min, crop_y_max
        if crop_active_val:
            m_width = res.get("ncols", 100)
            m_height = res.get("nrows", 100)
            col_c1, col_c2 = st.columns(2)
            c_xmin, c_xmax = col_c1.slider("X Pixel Crop Range", min_value=0, max_value=m_width, value=(0, m_width), key="quick_x_range")
            c_ymin, c_ymax = col_c2.slider("Y Pixel Crop Range", min_value=0, max_value=m_height, value=(0, m_height), key="quick_y_range")
            
        st.markdown("---")

        if analysis_method == "VCA (Unmixing)":
            n_em_all = res["n_endmembers"]
            labels_all = res["parsed_labels"]
            em_options_all = [labels_all[i] if (labels_all and i < len(labels_all)) else f"Endmember {i+1}" for i in range(n_em_all)]
            
            default_step2_ems = st.session_state.get("global_selected_endmembers", em_options_all)
            selected_ems_step2 = st.multiselect(
                "Select Endmembers to Display / Include in Downstream Analysis",
                em_options_all,
                default=default_step2_ems,
                key="step2_vca_select",
                on_change=sync_endmember_selection,
                args=("step2_vca_select",)
            )
            st.session_state["global_selected_endmembers"] = selected_ems_step2
            
            st.subheader("VCA Endmember Abundance Maps Grid")
            if not selected_ems_step2:
                st.warning("⚠️ No endmembers selected. Please select at least one endmember above.")
            else:
                ab_grid = res["abundances"]
                interp_grid = res["map_interpolation"]
                
                indices_grid = [em_options_all.index(name) for name in selected_ems_step2 if name in em_options_all]
                n_grid_sel = len(indices_grid)
                cols_grid = min(4, n_grid_sel)
                nrows_grid = (n_grid_sel + cols_grid - 1) // cols_grid
                fig_grid, axes_grid = plt.subplots(nrows_grid, cols_grid, figsize=(cols_grid * 4, nrows_grid * 3.5))
                axes_grid = np.atleast_1d(axes_grid).flatten()
                
                for idx_i, em_idx in enumerate(indices_grid):
                    m_oriented = crop_and_orient_map(ab_grid[:, :, em_idx], rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                    im_g = axes_grid[idx_i].imshow(m_oriented, cmap=map_cmap, interpolation=interp_grid)
                    lbl_text = em_options_all[em_idx]
                    axes_grid[idx_i].set_title(f"{lbl_text} Abundance", fontsize=10, fontweight="bold")
                    axes_grid[idx_i].axis("off")
                    fig_grid.colorbar(im_g, ax=axes_grid[idx_i], fraction=0.046, pad=0.04)
                for ax_g in axes_grid[n_grid_sel:]:
                    ax_g.axis("off")
                fig_grid.tight_layout()
                st.pyplot(fig_grid)
                plt.close(fig_grid)
                
            st.markdown("---")
            st.subheader("Interactive Component Zoom Viewer")
            
            n_em = res["n_endmembers"]
            abundances = res["abundances"]
            Ae = res["Ae"]
            wavenumber = res["wavenumber"]
            labels = res["parsed_labels"]
            interp_mode = res["map_interpolation"]
            
            comp_options = selected_ems_step2 if selected_ems_step2 else em_options_all
            selected_comp = st.selectbox("Select Component to Inspect", comp_options)
            comp_idx = em_options_all.index(selected_comp) if selected_comp in em_options_all else 0
            
            col_zoom1, col_zoom2 = st.columns(2)
            with col_zoom1:
                st.markdown(f"##### {selected_comp} Reference Spectrum")
                fig_spec, ax_spec = plt.subplots(figsize=(6, 4.5))
                wn_d = wrp._display_axis(wavenumber, res["skip_silent"])
                ax_spec.plot(wn_d, Ae[:, comp_idx], color="#1f77b4", lw=1.8)
                
                from scipy.signal import savgol_filter, find_peaks
                w_len = 15 if len(Ae[:, comp_idx]) > 15 else (len(Ae[:, comp_idx]) - 1 | 1)
                sm = savgol_filter(Ae[:, comp_idx], max(3, w_len), 3) if w_len >= 3 else Ae[:, comp_idx]
                pks, _ = find_peaks(sm, prominence=sm.max() * 0.05, distance=20)
                for p in pks[np.argsort(sm[pks])][-5:]:
                    ax_spec.text(wn_d[p], Ae[p, comp_idx] + 0.02 * Ae[:, comp_idx].max(), f"{wavenumber[p]:.0f}",
                                 color="#1f77b4", fontsize=8, fontweight="bold", ha="center")
                                 
                if res["skip_silent"]:
                    disp_t, orig_l = wrp._xticks_for_display(True)
                    ax_spec.set_xticks(disp_t); ax_spec.set_xticklabels(orig_l)
                ax_spec.set_xlabel("Wavenumber (cm-1)")
                ax_spec.set_ylabel("Intensity (a.u.)")
                ax_spec.grid(ls="--", alpha=0.3)
                fig_spec.tight_layout()
                st.pyplot(fig_spec)
                plt.close(fig_spec)
                
            with col_zoom2:
                st.markdown(f"##### {selected_comp} Abundance Map")
                fig_map, ax_map = plt.subplots(figsize=(6, 4.5))
                ab_comp_oriented = crop_and_orient_map(abundances[:, :, comp_idx], rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                im = ax_map.imshow(ab_comp_oriented, cmap=map_cmap, interpolation=interp_mode)
                ax_map.axis("off")
                fig_map.colorbar(im, ax=ax_map)
                fig_map.tight_layout()
                st.pyplot(fig_map)
                plt.close(fig_map)

        elif analysis_method == "PCA (Principal Components)":
            st.subheader("PCA 2D Score Maps Grid")
            pca_comp = res["pca_components"]
            scores = res["pca_scores"]
            pos_df = res["position"]
            m, n = res["ncols"], res["nrows"]
            
            cols_per_row = 3
            n_rows = (pca_comp + cols_per_row - 1) // cols_per_row
            for row_idx in range(n_rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    pc_idx = row_idx * cols_per_row + col_idx
                    if pc_idx < pca_comp:
                        pc_num = pc_idx + 1
                        score_vals = scores[pc_idx, :]
                        aux = np.full((m, n), np.nan)
                        for idx1 in range(scores.shape[1]):
                            try:
                                xi, yi = int(pos_df.iloc[idx1, 0]), int(pos_df.iloc[idx1, 1])
                                if 0 <= xi < m and 0 <= yi < n:
                                    aux[xi, yi] = score_vals[idx1]
                            except (ValueError, TypeError, IndexError):
                                pass
                                
                        if np.isnan(aux).all():
                            cols[col_idx].warning(f"PC {pc_num} Score Map: Spatial coordinates out of bounds.")
                            continue
                                
                        base_map = np.rot90(aux, 1, axes=(0, 1))
                        map_oriented = crop_and_orient_map(base_map, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                        fig_map, ax_map = plt.subplots(figsize=(4, 3.5))
                        im = ax_map.imshow(map_oriented, cmap="coolwarm", interpolation="nearest")
                        ax_map.set_title(f"PC {pc_num} Score Map", fontsize=10, fontweight="bold")
                        ax_map.axis("off")
                        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
                        fig_map.tight_layout()
                        cols[col_idx].pyplot(fig_map)
                        plt.close(fig_map)

        elif analysis_method == "HCA (Clustering)":
            st.subheader("HCA Cluster Spatial Map & Dendrogram Tree")
            col_hca1, col_hca2 = st.columns(2)
            with col_hca1:
                st.markdown("##### HCA Cluster Spatial Map")
                try:
                    colors = res["hca_obj"].show_map('auto', None, 1, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error rendering HCA map: {e}")
                plt.close('all')
            with col_hca2:
                st.markdown("##### HCA Dendrogram Tree")
                st.pyplot(res["hca_fig"])
                plt.close('all')

        elif analysis_method == "HDBSCAN":
            st.subheader("HDBSCAN Cluster Spatial Map")
            try:
                colors = res["hdb_obj"].show_map('auto', None, 1, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Error rendering HDBSCAN map: {e}")
            plt.close('all')

        # Spatial Map Relations Section
        st.markdown("---")
        st.subheader("🗺️ Spatial Map Relations & Composite Overlays")
        rel_tab1, rel_tab2, rel_tab3 = st.tabs(["3/4-Color Composite Overlay Map", "2D Component Spatial Ratio Map", "Merged Dominant Endmember Map"])
        
        with rel_tab1:
            st.markdown("##### Composite Co-localization Map (Colorblind-Friendly 3 & 4 Color Palettes)")
            overlay_modes_all = [
                "🔴🟢🔵 Standard 3-Channel RGB",
                "🔴🟢🔵🟡 Standard 4-Channel RGBY",
                "🌌 Okabe-Ito 3-Channel (Black/Orange/SkyBlue)",
                "🌌 Okabe-Ito 4-Channel (Black/Orange/SkyBlue/BluishGreen)",
                "🩵🩷🟨 Colorblind 3-Channel CMY",
                "🩵🩷🟨⬜ Colorblind 4-Channel CMYW",
                "👁️ Deuteranopia / Protanopia Safe",
                "👁️ Tritanopia Safe"
            ]
            overlay_mode = st.selectbox("Palette & Channel Mode", overlay_modes_all, key="vca_rgb_mode")
            
            PALETTE_CONFIGS = {
                "🔴🟢🔵 Standard 3-Channel RGB": {"n_chan": 3, "colors": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]},
                "🔴🟢🔵🟡 Standard 4-Channel RGBY": {"n_chan": 4, "colors": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]},
                "🌌 Okabe-Ito 3-Channel (Black/Orange/SkyBlue)": {"n_chan": 3, "colors": [(0.337, 0.706, 0.914), (0.902, 0.624, 0.0), (0.0, 0.620, 0.451)]},
                "🌌 Okabe-Ito 4-Channel (Black/Orange/SkyBlue/BluishGreen)": {"n_chan": 4, "colors": [(0.337, 0.706, 0.914), (0.902, 0.624, 0.0), (0.0, 0.620, 0.451), (0.800, 0.475, 0.655)]},
                "🩵🩷🟨 Colorblind 3-Channel CMY": {"n_chan": 3, "colors": [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]},
                "🩵🩷🟨⬜ Colorblind 4-Channel CMYW": {"n_chan": 4, "colors": [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]},
                "👁️ Deuteranopia / Protanopia Safe": {"n_chan": 4, "colors": [(0.337, 0.706, 0.914), (0.941, 0.894, 0.259), (0.835, 0.369, 0.0), (0.0, 0.620, 0.451)]},
                "👁️ Tritanopia Safe": {"n_chan": 3, "colors": [(0.800, 0.475, 0.655), (0.0, 0.620, 0.451), (0.835, 0.369, 0.0)]}
            }
            
            pal_cfg = PALETTE_CONFIGS.get(overlay_mode, PALETTE_CONFIGS["🔴🟢🔵 Standard 3-Channel RGB"])
            n_chan = pal_cfg["n_chan"]
            active_colors = pal_cfg["colors"]
            
            if analysis_method == "VCA (Unmixing)":
                n_em_rel = res["n_endmembers"]
                labels_rel = res["parsed_labels"]
                em_all_rel = [labels_rel[i] if (labels_rel and i < len(labels_rel)) else f"Endmember {i+1}" for i in range(n_em_rel)]
                rel_options = st.session_state.get("global_selected_endmembers", em_all_rel)
                if not rel_options: rel_options = em_all_rel
                ab_rel = res["abundances"]
                
                sel_channels = []
                cols_overlay = st.columns(n_chan)
                for c_i in range(n_chan):
                    prev_key_val = st.session_state.get(f"vca_overlay_c{c_i+1}")
                    s_idx = get_safe_index(prev_key_val, rel_options, c_i)
                    ch_sel = cols_overlay[c_i].selectbox(f"Ch {c_i+1}", rel_options, index=s_idx, key=f"vca_overlay_c{c_i+1}")
                    sel_channels.append(ch_sel)
                    
                ch_maps = [ab_rel[:, :, em_all_rel.index(ch_s)] for ch_s in sel_channels]
                comp_arr = synthesize_colocalization_overlay(ch_maps, active_colors)
                title_str = f"{n_chan}-Channel Overlay: " + " | ".join([f"Ch{i+1}={sel_channels[i]}" for i in range(n_chan)])
                
                comp_arr_oriented = crop_and_orient_map(comp_arr, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                
                fig_rgb, ax_rgb = plt.subplots(figsize=(7, 5))
                ax_rgb.imshow(comp_arr_oriented, interpolation=res["map_interpolation"])
                ax_rgb.set_title(title_str, fontsize=10, fontweight="bold")
                ax_rgb.axis("off")
                
                # Render Legend Patches
                legend_patches = []
                for idx, (color, name) in enumerate(zip(active_colors, sel_channels)):
                    patch = mpatches.Patch(color=color, label=f"Ch{idx+1}: {name}")
                    legend_patches.append(patch)
                ax_rgb.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
                
                fig_rgb.tight_layout()
                st.pyplot(fig_rgb)
                plt.close(fig_rgb)
                
            elif analysis_method == "PCA (Principal Components)":
                pca_comp_rel = res["pca_components"]
                rel_options = [f"PC {i+1}" for i in range(pca_comp_rel)]
                scores_rel = res["pca_scores"]
                pos_df_rel = res["position"]
                m_rel, n_rel = res["ncols"], res["nrows"]
                
                def get_pca_map(pc_i):
                    aux = np.full((m_rel, n_rel), 0.0)
                    for idx1 in range(scores_rel.shape[1]):
                        try:
                            xi, yi = int(pos_df_rel.iloc[idx1, 0]), int(pos_df_rel.iloc[idx1, 1])
                            if 0 <= xi < m_rel and 0 <= yi < n_rel:
                                aux[xi, yi] = scores_rel[pc_i, idx1]
                        except (ValueError, TypeError, IndexError):
                            pass
                    return np.rot90(aux, 1, axes=(0, 1))

                sel_channels = []
                cols_overlay = st.columns(n_chan)
                for c_i in range(n_chan):
                    prev_key_val = st.session_state.get(f"pca_overlay_c{c_i+1}")
                    s_idx = get_safe_index(prev_key_val, rel_options, c_i)
                    ch_sel = cols_overlay[c_i].selectbox(f"Ch {c_i+1}", rel_options, index=s_idx, key=f"pca_overlay_c{c_i+1}")
                    sel_channels.append(ch_sel)
                    
                ch_maps = [get_pca_map(rel_options.index(ch_s)) for ch_s in sel_channels]
                comp_arr = synthesize_colocalization_overlay(ch_maps, active_colors)
                title_str = f"PCA {n_chan}-Channel Overlay: " + " | ".join([f"Ch{i+1}={sel_channels[i]}" for i in range(n_chan)])
                
                comp_arr_oriented = crop_and_orient_map(comp_arr, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                
                fig_rgb, ax_rgb = plt.subplots(figsize=(7, 5))
                ax_rgb.imshow(comp_arr_oriented, interpolation="nearest")
                ax_rgb.set_title(title_str, fontsize=10, fontweight="bold")
                ax_rgb.axis("off")
                
                legend_patches = []
                for idx, (color, name) in enumerate(zip(active_colors, sel_channels)):
                    patch = mpatches.Patch(color=color, label=f"Ch{idx+1}: {name}")
                    legend_patches.append(patch)
                ax_rgb.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
                
                fig_rgb.tight_layout()
                st.pyplot(fig_rgb)
                plt.close(fig_rgb)

        with rel_tab2:
            st.markdown("##### 2D Component Spatial Ratio Map")
            col_rat1, col_rat2 = st.columns(2)
            if analysis_method == "VCA (Unmixing)":
                n_em_rel = res["n_endmembers"]
                labels_rel = res["parsed_labels"]
                em_all_rel = [labels_rel[i] if (labels_rel and i < len(labels_rel)) else f"Endmember {i+1}" for i in range(n_em_rel)]
                rel_options = st.session_state.get("global_selected_endmembers", em_all_rel)
                if not rel_options: rel_options = em_all_rel
                ab_rel = res["abundances"]
                
                s_num_idx = get_safe_index(st.session_state.get("vca_rat_num"), rel_options, 0)
                s_den_idx = get_safe_index(st.session_state.get("vca_rat_den"), rel_options, min(1, len(rel_options)-1))
                
                num_sel = col_rat1.selectbox("Numerator Component (A)", rel_options, index=s_num_idx, key="vca_rat_num")
                den_sel = col_rat2.selectbox("Denominator Component (B)", rel_options, index=s_den_idx, key="vca_rat_den")
                
                num_idx, den_idx = em_all_rel.index(num_sel), em_all_rel.index(den_sel)
                m_num = ab_rel[:, :, num_idx]
                m_den = ab_rel[:, :, den_idx]
                
                ratio_2d = m_num / np.where(m_den == 0, 1e-10, m_den)
                ratio_2d = np.clip(ratio_2d, 0, np.percentile(ratio_2d, 99))
                
                ratio_oriented = crop_and_orient_map(ratio_2d, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                
                fig_rat, ax_rat = plt.subplots(figsize=(6, 5))
                im_rat = ax_rat.imshow(ratio_oriented, cmap="viridis", interpolation=res["map_interpolation"])
                ax_rat.set_title(f"Spatial Ratio Map: {num_sel} / {den_sel}", fontsize=11, fontweight="bold")
                ax_rat.axis("off")
                fig_rat.colorbar(im_rat, ax=ax_rat, label="Ratio Value")
                fig_rat.tight_layout()
                st.pyplot(fig_rat)
                plt.close(fig_rat)
                
            elif analysis_method == "PCA (Principal Components)":
                pca_comp_rel = res["pca_components"]
                rel_options = [f"PC {i+1}" for i in range(pca_comp_rel)]
                
                s_num_idx = get_safe_index(st.session_state.get("pca_rat_num"), rel_options, 0)
                s_den_idx = get_safe_index(st.session_state.get("pca_rat_den"), rel_options, min(1, len(rel_options)-1))
                
                num_sel = col_rat1.selectbox("Numerator Component (A)", rel_options, index=s_num_idx, key="pca_rat_num")
                den_sel = col_rat2.selectbox("Denominator Component (B)", rel_options, index=s_den_idx, key="pca_rat_den")
                
                num_idx, den_idx = rel_options.index(num_sel), rel_options.index(den_sel)
                m_num = get_pca_map(num_idx)
                m_den = get_pca_map(den_idx)
                
                ratio_2d = m_num / np.where(m_den == 0, 1e-10, m_den)
                ratio_2d = np.clip(ratio_2d, np.percentile(ratio_2d, 1), np.percentile(ratio_2d, 99))
                
                ratio_oriented = crop_and_orient_map(ratio_2d, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                
                fig_rat, ax_rat = plt.subplots(figsize=(6, 5))
                im_rat = ax_rat.imshow(ratio_oriented, cmap="coolwarm", interpolation="nearest")
                ax_rat.set_title(f"PCA Spatial Ratio Map: {num_sel} / {den_sel}", fontsize=11, fontweight="bold")
                ax_rat.axis("off")
                fig_rat.colorbar(im_rat, ax=ax_rat, label="Ratio Value")
                fig_rat.tight_layout()
                st.pyplot(fig_rat)
                plt.close(fig_rat)

        with rel_tab3:
            st.markdown("##### Merged Dominant Endmember Map")
            st.markdown("Combines abundance maps into a single composite colormap where each pixel is colored by its dominant (highest abundance) endmember.")
            if analysis_method == "VCA (Unmixing)":
                n_em_rel = res["n_endmembers"]
                labels_rel = res["parsed_labels"]
                rel_options = [labels_rel[i] if (labels_rel and i < len(labels_rel)) else f"Endmember {i+1}" for i in range(n_em_rel)]
                ab_rel = res["abundances"]
                
                sel_merge_ems = st.multiselect("Select Endmembers to Include in Composite Merge", rel_options, default=st.session_state.get("global_selected_endmembers", rel_options), key="vca_merge_sel")
                
                if sel_merge_ems:
                    indices_merge = [rel_options.index(name) for name in sel_merge_ems]
                    ab_sub = ab_rel[:, :, indices_merge]
                    
                    dom_sub_idx = np.argmax(ab_sub, axis=-1)
                    dom_map = np.array([indices_merge[i] for i in dom_sub_idx.flatten()]).reshape(dom_sub_idx.shape)
                    
                    dom_oriented = crop_and_orient_map(dom_map, rotation=rot_val, flip_h=fliph_val, flip_v=flipv_val, crop_spatial=crop_active_val, crop_x_min=c_xmin, crop_x_max=c_xmax, crop_y_min=c_ymin, crop_y_max=c_ymax)
                    
                    import matplotlib.colors as mcolors
                    cmap_cat = plt.cm.get_cmap("tab10", n_em_rel)
                    norm_cat = mcolors.BoundaryNorm(np.arange(-0.5, n_em_rel + 0.5, 1), cmap_cat.N)
                    
                    fig_dom, ax_dom = plt.subplots(figsize=(7, 5.5))
                    im_dom = ax_dom.imshow(dom_oriented, cmap=cmap_cat, norm=norm_cat, interpolation=res["map_interpolation"])
                    ax_dom.set_title("Merged Dominant Endmember Map", fontsize=11, fontweight="bold")
                    ax_dom.axis("off")
                    
                    cbar_dom = fig_dom.colorbar(im_dom, ax=ax_dom, ticks=np.arange(n_em_rel), fraction=0.046, pad=0.04)
                    cbar_dom.ax.set_yticklabels(rel_options)
                    fig_dom.tight_layout()
                    st.pyplot(fig_dom)
                    plt.close(fig_dom)

        st.markdown("---")
        col_nav_p, col_nav_n = st.columns([1, 1])
        if col_nav_p.button("⬅️ Previous Step: Data Input & Selection", key="btn_step2_prev", use_container_width=True):
            update_workflow_step(0)
            st.rerun()
        if col_nav_n.button("Next Step: Reference Spectra & Peak Analysis ➡️", key="btn_step2_next", type="primary", use_container_width=True):
            update_workflow_step(2)
            st.rerun()

    # --------------------------------------------------------------------------
    # STEP 3: 📈 Reference Spectra & Peak Analysis
    # --------------------------------------------------------------------------
    elif current_step == 2:
        st.subheader("Step 3: 📈 Reference Spectra & Peak Analysis")
        if analysis_method == "VCA (Unmixing)":
            st.subheader("VCA Endmember Spectra Grid")
            n_em = res["n_endmembers"]
            Ae = res["Ae"]
            wavenumber = res["wavenumber"]
            labels = res["parsed_labels"]
            em_options = [labels[i] if (labels and i < len(labels)) else f"Endmember {i+1}" for i in range(n_em)]
            
            chosen_ems_step3 = st.session_state.get("global_selected_endmembers", em_options)
            selected_ems_grid = st.multiselect(
                "Select Endmembers to Display / Include in Analysis",
                em_options,
                default=chosen_ems_step3,
                key="step3_vca_grid_select",
                on_change=sync_endmember_selection,
                args=("step3_vca_grid_select",)
            )
            st.session_state["global_selected_endmembers"] = selected_ems_grid
            
            if selected_ems_grid:
                indices_spec = [em_options.index(name) for name in selected_ems_grid if name in em_options]
                cols_sp = min(4, len(indices_spec))
                nrows_sp = (len(indices_spec) + cols_sp - 1) // cols_sp
                fig_em_grid, axes_em_grid = plt.subplots(nrows_sp, cols_sp, figsize=(cols_sp * 3.5, nrows_sp * 2.5))
                axes_em_grid = np.atleast_1d(axes_em_grid).flatten()
                wn_d = wrp._display_axis(wavenumber, res["skip_silent"])
                
                for idx_i, em_idx in enumerate(indices_spec):
                    axes_em_grid[idx_i].plot(wn_d, Ae[:, em_idx], color="#1f77b4", lw=1.5)
                    axes_em_grid[idx_i].set_title(em_options[em_idx], fontsize=9, fontweight="bold")
                    axes_em_grid[idx_i].grid(ls="--", alpha=0.3)
                    if res["skip_silent"]:
                        disp_t, orig_l = wrp._xticks_for_display(True)
                        axes_em_grid[idx_i].set_xticks(disp_t); axes_em_grid[idx_i].set_xticklabels(orig_l, fontsize=7)
                for ax_sp in axes_em_grid[len(indices_spec):]:
                    ax_sp.axis("off")
                fig_em_grid.tight_layout()
                st.pyplot(fig_em_grid)
                plt.close(fig_em_grid)
            else:
                st.warning("⚠️ No endmembers selected.")
                
            st.markdown("---")
            st.subheader("Interactive Overlapped Endmember Spectra Plot")
            
            selected_ems_overlap = st.multiselect(
                "Select Endmembers for Overlapped Plot",
                em_options,
                default=st.session_state.get("global_selected_endmembers", em_options),
                key="step3_vca_overlap_select",
                on_change=sync_endmember_selection,
                args=("step3_vca_overlap_select",)
            )
            st.session_state["global_selected_endmembers"] = selected_ems_overlap
            
            col_p1, col_p2, col_p3 = st.columns(3)
            show_peaks = col_p1.checkbox("Find and Label Peaks", value=True, key="step3_vca_show_peaks")
            peak_prom = col_p2.slider("Peak Prominence", min_value=0.01, max_value=0.30, value=0.05, step=0.01, key="step3_vca_prom")
            y_offset = col_p3.slider("Y-Axis Offset", min_value=0.0, max_value=2.0, value=0.0, step=0.1, key="step3_vca_offset")
            
            if selected_ems_overlap:
                fig, ax = plt.subplots(figsize=(10, 5))
                wn_d = wrp._display_axis(wavenumber, res["skip_silent"])
                from scipy.signal import savgol_filter, find_peaks
                
                for idx_plot, name in enumerate(selected_ems_overlap):
                    idx = em_options.index(name)
                    spec = Ae[:, idx] / (np.max(Ae[:, idx]) or 1)
                    spec_offset = spec + (idx_plot * y_offset)
                    
                    line, = ax.plot(wn_d, spec_offset, label=name, lw=1.5)
                    
                    if show_peaks:
                        sm = savgol_filter(spec, 15, 3)
                        pks, _ = find_peaks(sm, prominence=sm.max() * peak_prom, distance=20)
                        for p in pks[np.argsort(sm[pks])][-5:]:
                            ax.text(wn_d[p], spec_offset[p] + 0.02, f"{wavenumber[p]:.0f}",
                                    color=line.get_color(), fontsize=8, fontweight="bold", ha="center")
                                    
                if res["skip_silent"]:
                    disp_t, orig_l = wrp._xticks_for_display(True)
                    ax.set_xticks(disp_t); ax.set_xticklabels(orig_l)
                ax.set_xlabel("Wavenumber (cm-1)")
                ax.set_ylabel("Normalised Intensity")
                ax.set_title("Overlapped Endmember Spectra (Normalised)", fontsize=12, fontweight="bold")
                ax.legend(frameon=True)
                ax.grid(ls="--", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif analysis_method == "PCA (Principal Components)":
            st.subheader("PCA Loadings Stacked Plot")
            pca_comp = res["pca_components"]
            loadings = res["pca_loadings"]
            wavenumber = res["wavenumber"]
            
            pc_options = [f"PC {i+1}" for i in range(pca_comp)]
            selected_pcs = st.multiselect(
                "Select PCs to Include in Loadings Stack",
                pc_options,
                default=st.session_state.get("global_selected_pcs", pc_options),
                key="step3_pca_select",
                on_change=sync_pc_selection,
                args=("step3_pca_select",)
            )
            st.session_state["global_selected_pcs"] = selected_pcs
            
            if selected_pcs:
                indices_pc = [pc_options.index(name) for name in selected_pcs if name in pc_options]
                fig_stack, axs = plt.subplots(len(indices_pc), sharex='all', sharey='all', figsize=(10, 1.8 * len(indices_pc)), gridspec_kw={'hspace': 0, 'left': 0.12, 'bottom': 0.15, 'right': 0.95, 'top': 0.9})
                axs = np.atleast_1d(axs).flatten()
                wn_num = pd.to_numeric(wavenumber)
                cmap = plt.cm.tab10(np.linspace(0, 1, max(10, pca_comp)))
                
                for idx_p, pc_i in enumerate(indices_pc):
                    color = cmap[pc_i % 10]
                    axs[idx_p].plot(wn_num, loadings[pc_i, :], color=color, lw=1.5, label=f"PC {pc_i+1}")
                    axs[idx_p].set_ylabel(f"PC {pc_i+1}", fontsize=9, fontweight="bold")
                    axs[idx_p].grid(ls="--", alpha=0.3)
                    axs[idx_p].axhline(0, color="gray", ls="--", alpha=0.5)
                    
                axs[-1].set_xlabel("Wavenumber (cm-1)", fontsize=10)
                axs[0].set_title("PCA Loadings Stacked Plot", fontsize=12, fontweight="bold")
                st.pyplot(fig_stack)
                plt.close(fig_stack)

        elif analysis_method in ["HCA (Clustering)", "HDBSCAN"]:
            st.subheader("Cluster Average Spectra")
            obj = res.get("hca_obj") or res.get("hdb_obj")
            wn = pd.to_numeric(obj.data.columns)
            unique_clusters = sorted(obj.label.unique())
            if len(unique_clusters) > 50:
                st.warning(f"Over 50 clusters detected ({len(unique_clusters)}). Displaying top 50 largest clusters in average spectra plot.")
                cluster_counts = obj.label.value_counts()
                top_clusters = cluster_counts.nlargest(50).index.tolist()
                unique_clusters = sorted(top_clusters)
            
            fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
            cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
            for c_idx, c_val in enumerate(unique_clusters):
                mask = obj.label == c_val
                cluster_data = obj.data[mask]
                if len(cluster_data) > 0:
                    mean_spec = cluster_data.mean(axis=0).values
                    std_spec = cluster_data.std(axis=0).values
                    color_c = cmap[c_idx % 10]
                    ax_avg.plot(wn, mean_spec, label=f"Cluster {c_val} (n={len(cluster_data)})", color=color_c, lw=1.8)
                    ax_avg.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, color=color_c, alpha=0.15)
            ax_avg.set_xlabel("Wavenumber (cm-1)")
            ax_avg.set_ylabel("Intensity (a.u.)")
            ax_avg.set_title("Cluster Mean Spectra (Mean ± Standard Deviation)", fontsize=12, fontweight="bold")
            ax_avg.legend(frameon=True)
            ax_avg.grid(ls="--", alpha=0.3)
            fig_avg.tight_layout()
            st.pyplot(fig_avg)
            plt.close(fig_avg)

        if res.get("run_use_glass") and res.get("glass_wn") is not None:
            st.markdown("---")
            st.subheader("Glass Background Spectrum Preview")
            fig_g, ax_g = plt.subplots(figsize=(10, 3.5))
            ax_g.plot(res["glass_wn"], res["glass_int"], color="gray", lw=1.5)
            ax_g.set_xlabel("Wavenumber (cm-1)")
            ax_g.set_ylabel("Intensity")
            ax_g.set_title("Glass Subtraction Background Spectrum", fontsize=11, fontweight="bold")
            ax_g.grid(ls="--", alpha=0.3)
            fig_g.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)

        st.markdown("---")
        col_nav_p, col_nav_n = st.columns([1, 1])
        if col_nav_p.button("⬅️ Previous Step: Spatial Mapping & Co-localization", key="btn_step3_prev", use_container_width=True):
            update_workflow_step(1)
            st.rerun()
        if col_nav_n.button("Next Step: Quantification & Downstream Statistics ➡️", key="btn_step3_next", type="primary", use_container_width=True):
            update_workflow_step(3)
            st.rerun()

    # --------------------------------------------------------------------------
    # STEP 4: 📊 Quantification & Downstream Statistics
    # --------------------------------------------------------------------------
    elif current_step == 3:
        st.subheader("Step 4: 📊 Quantification & Downstream Statistics")
        if analysis_method == "VCA (Unmixing)":
            st.subheader("Biochemical Quantification & Similarity")
            Ae = res["Ae"]
            wavenumber = res["wavenumber"]
            labels = res["parsed_labels"]
            n_em = res["n_endmembers"]
            em_names_all = [labels[i] if (labels and i < len(labels)) else f"Endmember {i+1}" for i in range(n_em)]
            
            chosen_ems = st.session_state.get("global_selected_endmembers", em_names_all)
            if not chosen_ems:
                st.warning("⚠️ No endmembers selected in global filter. Using all endmembers for quantification.")
                chosen_ems = em_names_all
                
            indices = [em_names_all.index(name) for name in chosen_ems if name in em_names_all]
            Ae_filtered = Ae[:, indices]
            em_names = [em_names_all[i] for i in indices]
            
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.markdown("##### Pearson Correlation Heatmap (Selected Endmembers)")
                if len(indices) == 0:
                    st.warning("No valid endmembers selected.")
                else:
                    corr_matrix = np.corrcoef(Ae_filtered.T)
                    if corr_matrix.ndim == 0:
                        corr_matrix = np.array([[1.0]])
                    fig_corr, ax_corr = plt.subplots(figsize=(5.5, 4.5))
                    im_corr = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
                    ax_corr.set_xticks(np.arange(len(em_names)))
                    ax_corr.set_yticks(np.arange(len(em_names)))
                    ax_corr.set_xticklabels(em_names, rotation=45, ha="right", fontsize=8)
                    ax_corr.set_yticklabels(em_names, fontsize=8)
                    for i in range(len(em_names)):
                        for j in range(len(em_names)):
                            ax_corr.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold",
                                         color="white" if abs(corr_matrix[i, j]) > 0.4 else "black")
                    ax_corr.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold")
                    fig_corr.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04)
                    fig_corr.tight_layout(pad=1.0)
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)
                
            with col_q2:
                st.markdown("##### Biochemical Macromolecular Peak Intensity Ratios")
                ratio_opt = [
                    "Lipid / Protein (I_2850 / I_2930)",
                    "Lipid Ester / Protein Amide I (I_1740 / I_1660)",
                    "Lipid / Protein Fingerprint (I_1440 / I_1660)",
                    "Protein Purity (I_1003 / I_1660)",
                    "DNA / Protein (I_785 / I_1003)"
                ]
                sel_ratio = st.selectbox("Select Biochemical Ratio Formula", ratio_opt)
                if sel_ratio == "Lipid / Protein (I_2850 / I_2930)": w1, w2 = 2850.0, 2930.0
                elif sel_ratio == "Lipid Ester / Protein Amide I (I_1740 / I_1660)": w1, w2 = 1740.0, 1660.0
                elif sel_ratio == "Lipid / Protein Fingerprint (I_1440 / I_1660)": w1, w2 = 1440.0, 1660.0
                elif sel_ratio == "Protein Purity (I_1003 / I_1660)": w1, w2 = 1003.0, 1660.0
                else: w1, w2 = 785.0, 1003.0
                
                idx1 = np.abs(wavenumber - w1).argmin()
                idx2 = np.abs(wavenumber - w2).argmin()
                diff1 = np.abs(wavenumber[idx1] - w1)
                diff2 = np.abs(wavenumber[idx2] - w2)
                if diff1 > 100 or diff2 > 100:
                    st.info(f"Using closest available channels: {wavenumber[idx1]:.0f} cm⁻¹ and {wavenumber[idx2]:.0f} cm⁻¹.")
                denom_safe = np.where(Ae_filtered[idx2, :] == 0, 1e-10, Ae_filtered[idx2, :])
                r_vals = Ae_filtered[idx1, :] / denom_safe
                r_vals = np.nan_to_num(r_vals, nan=0.0, posinf=0.0, neginf=0.0)
                
                fig_r, ax_r = plt.subplots(figsize=(5.5, 4.5))
                bars = ax_r.bar(em_names, r_vals, color="#1f77b4", edgecolor="black", alpha=0.85)
                ax_r.set_ylabel("Intensity Ratio Value")
                ax_r.set_title(f"Peak Ratio: {w1:.0f} / {w2:.0f} cm-1", fontsize=11, fontweight="bold")
                ax_r.set_xticklabels(em_names, rotation=45, ha="right", fontsize=8)
                for bar in bars:
                    h = bar.get_height()
                    ax_r.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                                  textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight="bold")
                ax_r.grid(axis="y", ls="--", alpha=0.3)
                fig_r.tight_layout(pad=1.0)
                st.pyplot(fig_r)
                plt.close(fig_r)

        elif analysis_method == "PCA (Principal Components)":
            st.subheader("PCA Score Scatter Plot (Synchronized Components)")
            pca_comp = res["pca_components"]
            scores = res["pca_scores"]
            pc_options_all = [f"PC {i+1}" for i in range(pca_comp)]
            chosen_pcs = st.session_state.get("global_selected_pcs", pc_options_all)
            if not chosen_pcs:
                chosen_pcs = pc_options_all
                
            col_s1, col_s2 = st.columns(2)
            s_x_idx = get_safe_index(st.session_state.get("step4_pc_x"), chosen_pcs, 0)
            s_y_idx = get_safe_index(st.session_state.get("step4_pc_y"), chosen_pcs, min(1, len(chosen_pcs) - 1))
            pc_x = col_s1.selectbox("X-axis Component", chosen_pcs, index=s_x_idx, key="step4_pc_x")
            pc_y = col_s2.selectbox("Y-axis Component", chosen_pcs, index=s_y_idx, key="step4_pc_y")
            idx_x = int(pc_x.split()[1]) - 1
            idx_y = int(pc_y.split()[1]) - 1
            
            fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
            ax_sc.scatter(scores[idx_x, :], scores[idx_y, :], alpha=0.7, color="#1f77b4", edgecolors="none")
            ax_sc.set_xlabel(pc_x)
            ax_sc.set_ylabel(pc_y)
            ax_sc.set_title(f"PCA Score Projection ({pc_x} vs {pc_y})", fontsize=12, fontweight="bold")
            ax_sc.grid(ls="--", alpha=0.3)
            fig_sc.tight_layout()
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        elif analysis_method in ["HCA (Clustering)", "HDBSCAN"]:
            st.subheader("Clustered Spectra Stack")
            obj = res.get("hca_obj") or res.get("hdb_obj")
            try:
                obj.show_stack(0.1, 0.5, 'auto')
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Error plotting stack: {e}")
            plt.close('all')

        st.markdown("---")
        col_nav_p, col_nav_n = st.columns([1, 1])
        if col_nav_p.button("⬅️ Previous Step: Reference Spectra & Peak Analysis", key="btn_step4_prev", use_container_width=True):
            update_workflow_step(2)
            st.rerun()
        if col_nav_n.button("Next Step: Export & Data Table Inspection ➡️", key="btn_step4_next", type="primary", use_container_width=True):
            update_workflow_step(4)
            st.rerun()

    # --------------------------------------------------------------------------
    # STEP 5: 💾 Export & Data Table Inspection
    # --------------------------------------------------------------------------
    elif current_step == 4:
        st.subheader("Step 5: 💾 Export & Data Table Inspection")
        
        st.markdown("##### Acquisition & Optics Metadata Settings")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown(f"**Output Directory**: `{custom_output_dir}`")
            if manual_export_btn:
                st.success(f"✅ Figures and CSV tables manually saved to `{custom_output_dir}`")
        with col_exp2:
            st.markdown(f"**Acquisition Metadata**: Wavelength=`{laser_wavelength or '532 nm'}` | Int. Time=`{integration_time}`s | Power=`{laser_power}`mW")
            
        st.markdown("---")
        st.subheader("Pipeline Processed Data Previews")
        
        if analysis_method == "VCA (Unmixing)":
            st.markdown("**Endmember Spectra Data Preview (First 50 Rows)**")
            st.dataframe(res["df_endmembers"].head(50))
            st.markdown("**Abundance Maps Data Preview (First 50 Rows)**")
            st.dataframe(res["df_abundances"].head(50))
        elif analysis_method == "PCA (Principal Components)":
            st.markdown("**PCA Scores Data Preview (First 50 Rows)**")
            st.dataframe(res["df_pca_scores"].head(50))
            st.markdown("**PCA Loadings Matrix Preview**")
            st.dataframe(res["df_pca_loadings"])
        elif analysis_method == "HCA (Clustering)":
            st.markdown("**HCA Clustering Data Preview (First 50 Rows)**")
            st.dataframe(res["df_hca"].head(50))
        elif analysis_method == "HDBSCAN":
            st.markdown("**HDBSCAN Clustering Data Preview (First 50 Rows)**")
            st.dataframe(res["df_hdbscan"].head(50))
            
        st.markdown("---")
        st.subheader("Pipeline JSON Run Metadata")
        st.json(res["metadata"])

        st.markdown("---")
        col_nav_p, _ = st.columns([1, 1])
        if col_nav_p.button("⬅️ Previous Step: Quantification & Downstream Statistics", key="btn_step5_prev", use_container_width=True):
            update_workflow_step(3)
            st.rerun()

else:
    st.info("📂 Please select or upload a dataset using the sidebar controls on the left to start processing.")
