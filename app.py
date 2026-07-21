import streamlit as st
import matplotlib.pyplot as plt
import os
import glob
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add tools to path so we can import witec_raman_pipeline
tools_dir = str(Path(__file__).parent / "tools")
if tools_dir not in sys.path:
    sys.path.append(tools_dir)
import witec_raman_pipeline as wrp
import importlib
importlib.reload(wrp)

from spectramap import spmap as sp
from smart_importer import parse_with_ollama

st.set_page_config(page_title="SpectraMap GUI", layout="wide")

st.title("SpectraMap GUI")

# Mode selector
app_mode = st.sidebar.selectbox("App Mode", ["WITec Raman Pipeline", "General Analysis"])

# Helper function to get data files
def get_data_files():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(data_dir):
        files = glob.glob(os.path.join(data_dir, '*.csv.xz')) + glob.glob(os.path.join(data_dir, '*.spc'))
        return [os.path.basename(f) for f in files]
    return []

def save_uploaded_file(uploaded_file, filename):
    temp_dir = Path(__file__).parent / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)

if app_mode == "General Analysis":
    # Sidebar for controls
    st.sidebar.header("Data Loading")

    load_mode = st.sidebar.radio("Load Mode", ["Sample Datasets", "Smart Importer (AI)"])

    if 'sp_obj' not in st.session_state:
        st.session_state.sp_obj = None

    if load_mode == "Sample Datasets":
        data_files = get_data_files()
        if data_files:
            selected_file = st.sidebar.selectbox("Select a sample dataset", ["-- Select --"] + data_files)
        else:
            st.sidebar.error("No data files found in 'data/' folder.")
            selected_file = "-- Select --"

        data_type = st.sidebar.selectbox("Data Type", ['hyper_image', 'multi_spectra', 'single_spectrum'])

        if selected_file != "-- Select --":
            if st.sidebar.button("Load Data"):
                file_path = os.path.join(os.path.dirname(__file__), 'data', selected_file)
                if selected_file.endswith('.csv.xz'):
                    base_path = file_path[:-7] 
                elif selected_file.endswith('.spc'):
                    base_path = file_path[:-4]
                else:
                    base_path = file_path

                with st.spinner("Loading data..."):
                    st.session_state.sp_obj = sp.hyper_object(selected_file.split('.')[0], data_type=data_type)
                    if selected_file.endswith('.csv.xz'):
                        try:
                            st.session_state.sp_obj.read_csv_xz(base_path)
                        except Exception as e:
                            if 'z' in str(e):
                                st.session_state.sp_obj.read_csv_3d_xz(base_path)
                            else:
                                raise e
                    elif selected_file.endswith('.spc'):
                        st.session_state.sp_obj.read_spc(base_path)
                    st.sidebar.success("Data loaded successfully!")
                    import copy
                    st.session_state.sp_obj_raw = copy.deepcopy(st.session_state.sp_obj)

    elif load_mode == "Smart Importer (AI)":
        uploaded_file = st.sidebar.file_uploader("Upload custom data file")
        data_type = st.sidebar.selectbox("Data Type", ['hyper_image', 'multi_spectra', 'single_spectrum'])
        
        if uploaded_file:
            if st.sidebar.button("Analyze & Load with AI"):
                with st.spinner("AI is analyzing your file (via Ollama)..."):
                    try:
                        df, code = parse_with_ollama(uploaded_file.getvalue(), uploaded_file.name)
                        st.sidebar.success("AI Successfully parsed the data!")
                        with st.sidebar.expander("Show AI Generated Code"):
                            st.code(code, language='python')
                            
                        obj = sp.hyper_object(uploaded_file.name.split('.')[0], data_type=data_type)
                        
                        obj.data = df.drop(columns=['label', 'x', 'y', 'z'], errors='ignore')
                        if 'label' in df.columns:
                            obj.label = pd.Series(df['label'])
                        else:
                            obj.label = pd.Series([1]*len(df))
                            
                        if 'x' in df.columns and 'y' in df.columns:
                            if 'z' in df.columns:
                                obj.position = df[['x', 'y', 'z']]
                            else:
                                obj.position = df[['x', 'y']]
                        else:
                            obj.position = pd.DataFrame({'x': np.arange(len(df)), 'y': np.zeros(len(df))})
                            
                        obj.m = int(pd.to_numeric(obj.position['x']).max() + 1) if 'x' in obj.position else len(df)
                        obj.n = int(pd.to_numeric(obj.position['y']).max() + 1) if 'y' in obj.position else 1
                        obj.resolution = 1
                        obj.sublabel = pd.Series(np.zeros(len(obj.data)), name="sublabel")
                        
                        st.session_state.sp_obj = obj
                        import copy
                        st.session_state.sp_obj_raw = copy.deepcopy(obj)
                    except Exception as e:
                        st.sidebar.error(str(e))

    if st.session_state.sp_obj is not None:
        obj = st.session_state.sp_obj
        st.sidebar.markdown("---")
        st.sidebar.header("Preprocessing")
        
        col_keep = st.sidebar.columns(2)
        keep_min = col_keep[0].number_input("Keep Min", value=400)
        keep_max = col_keep[1].number_input("Keep Max", value=1850)
        if st.sidebar.button("Apply Keep"):
            obj.keep(keep_min, keep_max)
            st.sidebar.success(f"Kept {keep_min} to {keep_max}")

        snip_iter = st.sidebar.number_input("SNIP iterations", value=30, min_value=1)
        if st.sidebar.button("Apply SNIP"):
            with st.spinner("Applying SNIP..."):
                obj.snip(snip_iter)
            st.sidebar.success(f"SNIP applied ({snip_iter} iterations)")
            
        gaussian_sigma = st.sidebar.number_input("Gaussian Sigma", value=2, min_value=1)
        if st.sidebar.button("Apply Gaussian"):
            obj.gaussian(gaussian_sigma)
            st.sidebar.success(f"Gaussian applied (sigma {gaussian_sigma})")
            
        if st.sidebar.button("Apply Vector Normalization"):
            obj.vector()
            st.sidebar.success("Vector normalization applied")

        st.sidebar.markdown("##### Spatial Smoothing")
        smooth_spatial_sigma = st.sidebar.number_input("Spatial Gaussian Sigma (0.0 to disable)", min_value=0.0, value=0.0, step=0.1, help="Applies a 2D Gaussian filter spatially over the pixels for each channel.")
        if st.sidebar.button("Apply Spatial Gaussian"):
            if smooth_spatial_sigma > 0:
                with st.spinner("Applying Spatial Gaussian..."):
                    matrix = obj.data.values.T
                    matrix_smoothed = wrp.spatial_gaussian_smooth(matrix, obj.m, obj.n, smooth_spatial_sigma)
                    obj.data = pd.DataFrame(matrix_smoothed.T, index=obj.data.index, columns=obj.data.columns)
                st.sidebar.success(f"Spatial Gaussian applied (sigma {smooth_spatial_sigma})")

        st.sidebar.markdown("---")
        st.sidebar.header("Analysis")
        
        analysis_type = st.sidebar.selectbox("Analysis", ["None", "HDBSCAN", "PCA", "HCA"])
        
        if analysis_type == "HDBSCAN":
            hdb_min_cluster_size = st.sidebar.number_input("Min Cluster Size", value=5, min_value=2)
            hdb_min_samples = st.sidebar.number_input("Min Samples", value=5, min_value=1)
            if st.sidebar.button("Run HDBSCAN"):
                with st.spinner("Running HDBSCAN..."):
                    obj.hdbscan(hdb_min_cluster_size, hdb_min_samples)
                st.session_state.colors = 'auto'
                st.sidebar.success("HDBSCAN complete")
                
        elif analysis_type == "PCA":
            pca_components = st.sidebar.number_input("PCA Components", value=3, min_value=1)
            if st.sidebar.button("Run PCA"):
                with st.spinner("Running PCA..."):
                    scores, loadings = obj.pca(pca_components, False)
                    st.session_state.scores = scores
                    st.session_state.loadings = loadings
                st.sidebar.success("PCA complete")
                
        elif analysis_type == "HCA":
            hca_distance = st.sidebar.selectbox("Distance Metric", ["euclidean", "cosine", "manhattan", "pearson"])
            
            if hca_distance in ["pearson", "cosine", "manhattan"]:
                hca_linkage = st.sidebar.selectbox("Linkage Method", ["complete", "average", "single"])
            else:
                hca_linkage = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
                
            hca_dist = st.sidebar.number_input("Distance Threshold (dist)", min_value=0.0, value=1.0, step=0.1, help="Cut-off threshold for dendrogram branches")
            
            truncate_dendrogram = st.sidebar.checkbox("Truncate Dendrogram View", value=False)
            truncate_p_val = None
            if truncate_dendrogram:
                truncate_p_val = st.sidebar.number_input("Number of Branches (p)", min_value=2, value=10, step=1)
                
            if st.sidebar.button("Run HCA"):
                with st.spinner("Running HCA..."):
                    plt.close('all') # Clear existing plots
                    obj.hca(hca_distance, hca_linkage, hca_dist, truncate_p_val)
                    st.session_state.hca_fig = plt.gcf()
                    st.session_state.colors = 'auto'
                st.sidebar.success("HCA complete")

        st.sidebar.markdown("---")
        st.sidebar.header("Export Results")
        gen_output_dir = st.sidebar.text_input("Output Directory Path", value="./export_general", help="Path to save all processed data, CSVs, figures, and metadata.")
        
        if st.sidebar.button("Save General Analysis Data"):
            try:
                out_path = Path(gen_output_dir)
                fig_path = out_path / "figures"
                proc_path = out_path / "processed"
                
                fig_path.mkdir(parents=True, exist_ok=True)
                proc_path.mkdir(parents=True, exist_ok=True)
                
                # 1. Save Preprocessed Spectra
                spectra_csv = proc_path / "preprocessed_spectra.csv"
                obj.data.to_csv(spectra_csv)
                
                # 2. Save Positions & Labels
                if obj.position is not None:
                    obj.position.to_csv(proc_path / "positions.csv", index=False)
                if obj.label is not None:
                    obj.label.to_csv(proc_path / "cluster_labels.csv", index=False, header=["label"])
                    
                # 3. Save Metadata JSON
                metadata = {
                    "DATA_NAME": obj.name,
                    "DATA_TYPE": obj.data_type,
                    "M_ROWS": obj.m,
                    "N_COLS": obj.n,
                    "ANALYSIS_TYPE": analysis_type,
                    "PREPROCESSING": {
                        "KEEP_MIN": keep_min,
                        "KEEP_MAX": keep_max,
                        "SNIP_ITERATIONS": snip_iter,
                        "GAUSSIAN_SIGMA": gaussian_sigma,
                        "SPATIAL_GAUSSIAN_SIGMA": smooth_spatial_sigma
                    }
                }
                
                # 4. Save analysis-specific data & figures
                if analysis_type == "PCA" and st.session_state.get('scores'):
                    scores = st.session_state.scores
                    loadings = st.session_state.loadings
                    
                    scores.data.to_csv(proc_path / "pca_scores.csv")
                    loadings.data.to_csv(proc_path / "pca_loadings.csv")
                    
                    metadata["PCA_PARAMS"] = {
                        "COMPONENTS": pca_components
                    }
                    
                    # Generate and save PCA figures
                    # a. Score maps for all components
                    for i in range(len(scores.label)):
                        pc_num = i + 1
                        score_vals = scores.data.iloc[i].values
                        aux = np.zeros((obj.m, obj.n))
                        aux[:] = np.nan
                        for idx1 in range(len(obj.data.index)):
                            xi = int(pd.to_numeric(obj.position.iloc[idx1, 0]))
                            yi = int(pd.to_numeric(obj.position.iloc[idx1, 1]))
                            if 0 <= xi < obj.m and 0 <= yi < obj.n:
                                aux[xi, yi] = score_vals[idx1]
                        
                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(np.rot90(aux, 1, axes=(0, 1)), cmap="coolwarm", interpolation="nearest")
                        ax.set_title(f"PC {pc_num} Score Map", fontsize=11, fontweight="bold")
                        ax.axis("off")
                        plt.colorbar(im, ax=ax)
                        fig.tight_layout()
                        fig.savefig(str(fig_path / f"pca_score_map_pc{pc_num}.png"), dpi=150)
                        plt.close(fig)
                        
                    # b. Loadings spectrum for all components
                    wn = pd.to_numeric(loadings.wavenumber)
                    for i in range(len(loadings.label)):
                        pc_num = i + 1
                        load_vals = loadings.data.iloc[i].values
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(wn, load_vals, color="#2ca02c", lw=1.8)
                        ax.set_title(f"PC {pc_num} Loadings", fontsize=11, fontweight="bold")
                        ax.set_xlabel("Wavenumber (cm-1)")
                        ax.set_ylabel("Loading Weight")
                        ax.set_xlim(wn.min(), wn.max())
                        ax.grid(ls="--", alpha=0.3)
                        fig.tight_layout()
                        fig.savefig(str(fig_path / f"pca_loadings_pc{pc_num}.png"), dpi=150)
                        plt.close(fig)
                        
                    # c. PCA scatter
                    fig, ax = plt.subplots(figsize=(6, 5))
                    x_vals = scores.data.iloc[0].values
                    y_vals = scores.data.iloc[min(1, len(scores.label)-1)].values
                    unique_labels = obj.label.unique()
                    if len(unique_labels) > 1 and len(unique_labels) <= 20:
                        for lbl in unique_labels:
                            mask = obj.label == lbl
                            ax.scatter(x_vals[mask], y_vals[mask], label=str(lbl), alpha=0.7, edgecolors="none")
                        ax.legend()
                    else:
                        ax.scatter(x_vals, y_vals, alpha=0.7, color="#1f77b4", edgecolors="none")
                    ax.set_xlabel("PC 1")
                    ax.set_ylabel(f"PC {min(2, len(scores.label))}")
                    ax.set_title("PCA Score scatter plot", fontsize=11, fontweight="bold")
                    ax.grid(ls="--", alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(str(fig_path / "pca_scatter.png"), dpi=150)
                    plt.close(fig)
                    
                elif analysis_type == "HCA" and st.session_state.get('hca_fig'):
                    metadata["HCA_PARAMS"] = {
                        "DISTANCE": hca_distance,
                        "LINKAGE": hca_linkage,
                        "THRESHOLD": hca_dist,
                        "TRUNCATED": truncate_dendrogram,
                        "TRUNCATE_P": truncate_p_val
                    }
                    # Save dendrogram
                    st.session_state.hca_fig.savefig(str(fig_path / "hca_dendrogram.png"), dpi=150)
                    
                    # Save map plot
                    fig_map, ax_map = plt.subplots(figsize=(6, 5))
                    try:
                        colors = obj.show_map('auto', None, 1)
                        plt.savefig(str(fig_path / "hca_cluster_map.png"), dpi=150)
                    except Exception as e:
                        print(f"HCA map export error: {e}")
                    plt.close('all')
                    
                    # Save stack plot
                    fig_stack, ax_stack = plt.subplots(figsize=(10, 5))
                    try:
                        obj.show_stack(0.1, 0.5, 'auto')
                        plt.savefig(str(fig_path / "hca_spectra_stack.png"), dpi=150)
                    except Exception as e:
                        print(f"HCA stack export error: {e}")
                    plt.close('all')
                    
                    # Save cluster average spectra
                    wn = pd.to_numeric(obj.data.columns)
                    unique_clusters = sorted(obj.label.unique())
                    fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
                    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
                    for c_idx, c_val in enumerate(unique_clusters):
                        mask = obj.label == c_val
                        cluster_data = obj.data[mask]
                        if len(cluster_data) > 0:
                            mean_spec = cluster_data.mean(axis=0).values
                            std_spec = cluster_data.std(axis=0).values
                            color_c = cmap[c_idx % 10]
                            line, = ax_avg.plot(wn, mean_spec, label=f"Cluster {c_val} (n={len(cluster_data)})", color=color_c, lw=1.8)
                            ax_avg.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, color=color_c, alpha=0.15)
                    ax_avg.set_xlabel("Wavenumber (cm-1)")
                    ax_avg.set_ylabel("Intensity (a.u.)")
                    ax_avg.set_xlim(wn.min(), wn.max())
                    ax_avg.set_title("Cluster Mean Spectra (Mean ± Standard Deviation)", fontsize=11, fontweight="bold")
                    ax_avg.legend(frameon=True)
                    ax_avg.grid(ls="--", alpha=0.3)
                    fig_avg.tight_layout()
                    fig_avg.savefig(str(fig_path / "hca_cluster_average_spectra.png"), dpi=150)
                    plt.close(fig_avg)
                    
                elif analysis_type == "HDBSCAN" and st.session_state.get('colors'):
                    metadata["HDBSCAN_PARAMS"] = {
                        "MIN_CLUSTER_SIZE": hdb_min_cluster_size,
                        "MIN_SAMPLES": hdb_min_samples
                    }
                    
                    # Recreate and save HDBSCAN map
                    fig_map, ax_map = plt.subplots(figsize=(6, 5))
                    try:
                        colors = obj.show_map('auto', None, 1)
                        plt.savefig(str(fig_path / "hdbscan_cluster_map.png"), dpi=150)
                    except Exception as e:
                        print(f"HDBSCAN map export error: {e}")
                    plt.close('all')
                    
                    # Recreate and save HDBSCAN stack
                    fig_stack, ax_stack = plt.subplots(figsize=(10, 5))
                    try:
                        obj.show_stack(0.1, 0.5, 'auto')
                        plt.savefig(str(fig_path / "hdbscan_spectra_stack.png"), dpi=150)
                    except Exception as e:
                        print(f"HDBSCAN stack export error: {e}")
                    plt.close('all')
                
                with open(out_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
                    
                st.sidebar.success(f"Results successfully saved to: `{out_path}`")
            except Exception as e:
                st.sidebar.error(f"Failed to export results: {e}")

        # Main view
        st.header("Visualization")
        
        if analysis_type == "HDBSCAN" and st.session_state.get('colors'):
            tab_map, tab_stack = st.tabs(["Cluster Map", "Spectra Stack"])
            
            with tab_map:
                st.subheader("HDBSCAN Cluster Map")
                try:
                    colors = obj.show_map('auto', None, 1)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error plotting map: {e}")
                plt.close('all')
                
            with tab_stack:
                st.subheader("HDBSCAN Clustered Spectra Stack")
                try:
                    obj.show_stack(0.1, 0.5, 'auto')
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error plotting stack: {e}")
                plt.close('all')
                
        elif analysis_type == "PCA" and st.session_state.get('scores'):
            scores = st.session_state.scores
            loadings = st.session_state.loadings
            num_components = len(scores.label)
            
            tab_map, tab_load, tab_scat = st.tabs(["PCA Score Maps", "PCA Loadings", "PCA Scatter"])
            
            with tab_map:
                st.subheader("PCA 2D Score Maps Grid")
                cols_per_row = 3
                n_rows = (num_components + cols_per_row - 1) // cols_per_row
                for row_idx in range(n_rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        pc_idx = row_idx * cols_per_row + col_idx
                        if pc_idx < num_components:
                            pc_num = pc_idx + 1
                            score_vals = scores.data.iloc[pc_idx].values
                            
                            aux = np.zeros((obj.m, obj.n))
                            aux[:] = np.nan
                            for idx1 in range(len(obj.data.index)):
                                xi = int(pd.to_numeric(obj.position.iloc[idx1, 0]))
                                yi = int(pd.to_numeric(obj.position.iloc[idx1, 1]))
                                if 0 <= xi < obj.m and 0 <= yi < obj.n:
                                    aux[xi, yi] = score_vals[idx1]
                                    
                            fig_map, ax_map = plt.subplots(figsize=(4, 3.5))
                            im = ax_map.imshow(np.rot90(aux, 1, axes=(0, 1)), cmap="coolwarm", interpolation="nearest")
                            ax_map.set_title(f"PC {pc_num} Score Map", fontsize=10, fontweight="bold")
                            ax_map.axis("off")
                            plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
                            fig_map.tight_layout()
                            cols[col_idx].pyplot(fig_map)
                            plt.close(fig_map)
                
            with tab_load:
                st.subheader("PCA Loadings Spectrum")
                pc_selection_l = st.selectbox("Select Principal Component for Loadings", [f"PC {i}" for i in range(1, num_components + 1)], key="pca_load_pc")
                pc_idx_l = int(pc_selection_l.split()[1]) - 1
                
                wn = pd.to_numeric(loadings.wavenumber)
                load_vals = loadings.data.iloc[pc_idx_l].values
                
                col_pl1, col_pl2 = st.columns(2)
                show_peaks = col_pl1.checkbox("Find and Label Peaks", value=True, key="pca_loadings_show_peaks")
                peak_prominence = col_pl2.slider("Peak Prominence (Fraction of max)", min_value=0.01, max_value=0.30, value=0.05, step=0.01, key="pca_loadings_prominence")
                
                fig_load, ax_load = plt.subplots(figsize=(10, 4.5))
                ax_load.plot(wn, load_vals, color="#2ca02c", lw=1.8)
                
                if show_peaks:
                    from scipy.signal import find_peaks
                    max_val = np.max(np.abs(load_vals)) or 1
                    norm_vals = load_vals / max_val
                    
                    pks_pos, _ = find_peaks(norm_vals, prominence=peak_prominence, distance=20)
                    pks_neg, _ = find_peaks(-norm_vals, prominence=peak_prominence, distance=20)
                    
                    for p in pks_pos:
                        ax_load.text(wn.iloc[p], load_vals[p] + 0.02 * max_val, f"{wn.iloc[p]:.0f}",
                                     color="blue", fontsize=8, fontweight="bold", ha="center")
                    for p in pks_neg:
                        ax_load.text(wn.iloc[p], load_vals[p] - 0.04 * max_val, f"{wn.iloc[p]:.0f}",
                                     color="red", fontsize=8, fontweight="bold", ha="center")
                                     
                ax_load.set_title(f"{pc_selection_l} Loadings Vector", fontsize=12, fontweight="bold")
                ax_load.set_xlabel("Wavenumber (cm-1)")
                ax_load.set_ylabel("Loading Weight")
                ax_load.set_xlim(wn.min(), wn.max())
                ax_load.grid(ls="--", alpha=0.3)
                fig_load.tight_layout()
                st.pyplot(fig_load)
                plt.close(fig_load)
                
            with tab_scat:
                st.subheader("PCA Score Scatter Plot")
                col_sc1, col_sc2 = st.columns(2)
                pc_x = col_sc1.selectbox("X-axis Component", [f"PC {i}" for i in range(1, num_components + 1)], index=0, key="pca_scat_x")
                pc_y = col_sc2.selectbox("Y-axis Component", [f"PC {i}" for i in range(1, num_components + 1)], index=min(1, num_components - 1), key="pca_scat_y")
                
                idx_x = int(pc_x.split()[1]) - 1
                idx_y = int(pc_y.split()[1]) - 1
                
                fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
                x_vals = scores.data.iloc[idx_x].values
                y_vals = scores.data.iloc[idx_y].values
                
                unique_labels = obj.label.unique()
                if len(unique_labels) > 1 and len(unique_labels) <= 20:
                    for lbl in unique_labels:
                        mask = obj.label == lbl
                        ax_sc.scatter(x_vals[mask], y_vals[mask], label=str(lbl), alpha=0.7, edgecolors="none")
                    ax_sc.legend()
                else:
                    ax_sc.scatter(x_vals, y_vals, alpha=0.7, color="#1f77b4", edgecolors="none")
                    
                ax_sc.set_xlabel(pc_x)
                ax_sc.set_ylabel(pc_y)
                ax_sc.set_title(f"PCA Score Projection ({pc_x} vs {pc_y})", fontsize=12, fontweight="bold")
                ax_sc.grid(ls="--", alpha=0.3)
                fig_sc.tight_layout()
                st.pyplot(fig_sc)
                plt.close(fig_sc)
            
        elif analysis_type == "HCA" and st.session_state.get('hca_fig'):
            tab_dend, tab_map, tab_stack, tab_avg = st.tabs(["Dendrogram (Tree)", "Cluster Map", "Spectra Stack", "Cluster Average Spectra"])
            
            with tab_dend:
                st.subheader("HCA Dendrogram Tree")
                st.pyplot(st.session_state.hca_fig)
                
            with tab_map:
                st.subheader("HCA Cluster Map (Sections)")
                try:
                    colors = obj.show_map('auto', None, 1)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error plotting map: {e}")
                plt.close('all')
                
            with tab_stack:
                st.subheader("HCA Clustered Spectra Stack")
                try:
                    obj.show_stack(0.1, 0.5, 'auto')
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error plotting stack: {e}")
                plt.close('all')
                
            with tab_avg:
                st.subheader("HCA Cluster Average Spectra")
                wn = pd.to_numeric(obj.data.columns)
                unique_clusters = sorted(obj.label.unique())
                
                col_ap1, col_ap2 = st.columns(2)
                show_pks_avg = col_ap1.checkbox("Find and Label Cluster Peaks", value=True, key="hca_avg_peaks")
                prom_avg = col_ap2.slider("Peak Prominence (Fraction of max)", min_value=0.01, max_value=0.30, value=0.05, step=0.01, key="hca_avg_prom")
                
                fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
                cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
                
                from scipy.signal import savgol_filter, find_peaks
                
                for c_idx, c_val in enumerate(unique_clusters):
                    mask = obj.label == c_val
                    cluster_data = obj.data[mask]
                    
                    if len(cluster_data) > 0:
                        mean_spec = cluster_data.mean(axis=0).values
                        std_spec = cluster_data.std(axis=0).values
                        
                        color_c = cmap[c_idx % 10]
                        line, = ax_avg.plot(wn, mean_spec, label=f"Cluster {c_val} (n={len(cluster_data)})", color=color_c, lw=1.8)
                        ax_avg.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, color=color_c, alpha=0.15)
                        
                        if show_pks_avg:
                            window_len = 15
                            if window_len >= len(mean_spec):
                                window_len = len(mean_spec) - 1
                                if window_len % 2 == 0:
                                    window_len -= 1
                            if window_len >= 3:
                                sm = savgol_filter(mean_spec, window_len, 3)
                            else:
                                sm = mean_spec
                                
                            pks, _ = find_peaks(sm, prominence=sm.max() * prom_avg, distance=20)
                            for p in pks[np.argsort(sm[pks])][-5:]:
                                ax_avg.text(wn.iloc[p], mean_spec[p] + 0.02 * mean_spec.max(), f"{wn.iloc[p]:.0f}",
                                            color=color_c, fontsize=8, fontweight="bold", ha="center")
                                            
                ax_avg.set_xlabel("Wavenumber (cm-1)")
                ax_avg.set_ylabel("Intensity (a.u.)")
                ax_avg.set_xlim(wn.min(), wn.max())
                ax_avg.set_title("Cluster Mean Spectra (Mean ± Standard Deviation)", fontsize=12, fontweight="bold")
                ax_avg.legend(frameon=True)
                ax_avg.grid(ls="--", alpha=0.3)
                fig_avg.tight_layout()
                st.pyplot(fig_avg)
                plt.close(fig_avg)
            
        else:
            st.header("Data Inspector & Preprocessing Preview")
            if st.session_state.get("sp_obj_raw") is not None:
                obj_raw = st.session_state.sp_obj_raw
                n_pixels = len(obj.data)
                
                if obj.data_type == 'hyper_image':
                    st.markdown(f"**Dataset Dimensions:** `{obj.m}` x `{obj.n}` pixels ({n_pixels} total), `{len(obj.data.columns)}` wavenumber channels.")
                    col_x, col_y = st.columns(2)
                    sel_x = col_x.slider("Inspect Pixel X Coordinate", min_value=0, max_value=obj.m-1, value=0, key="inspect_x")
                    sel_y = col_y.slider("Inspect Pixel Y Coordinate", min_value=0, max_value=obj.n-1, value=0, key="inspect_y")
                    
                    match = obj.position[(pd.to_numeric(obj.position['x']) == sel_x) & (pd.to_numeric(obj.position['y']) == sel_y)]
                    if not match.empty:
                        pixel_idx = match.index[0]
                    else:
                        pixel_idx = 0
                else:
                    st.markdown(f"**Dataset Size:** `{n_pixels}` spectra, `{len(obj.data.columns)}` wavenumber channels.")
                    pixel_idx = st.slider("Inspect Spectrum Index", min_value=0, max_value=n_pixels-1, value=0, key="inspect_idx")
                    
                fig_prev, ax_prev = plt.subplots(figsize=(10, 4.5))
                
                raw_y = obj_raw.data.iloc[pixel_idx].values
                raw_x = pd.to_numeric(obj_raw.data.columns).values
                ax_prev.plot(raw_x, raw_y, label="Raw Spectrum", color="gray", alpha=0.7, ls="--")
                
                proc_y = obj.data.iloc[pixel_idx].values
                proc_x = pd.to_numeric(obj.data.columns).values
                
                is_preprocessed = (len(raw_x) != len(proc_x)) or not np.allclose(raw_y, proc_y, equal_nan=True)
                
                if is_preprocessed:
                    ax_prev.plot(proc_x, proc_y, label="Preprocessed Spectrum", color="#1f77b4", lw=1.8)
                    ax_prev.set_title(f"Spectrum Inspector (Index {pixel_idx}) — Preprocessing Applied", fontsize=12, fontweight="bold")
                else:
                    ax_prev.set_title(f"Spectrum Inspector (Index {pixel_idx}) — Raw Data", fontsize=12, fontweight="bold")
                    
                ax_prev.set_xlabel("Wavenumber (cm-1)")
                ax_prev.set_ylabel("Intensity (a.u.)")
                ax_prev.set_xlim(min(raw_x.min(), proc_x.min()), max(raw_x.max(), proc_x.max()))
                ax_prev.legend()
                ax_prev.grid(ls="--", alpha=0.3)
                fig_prev.tight_layout()
                st.pyplot(fig_prev)
                plt.close(fig_prev)
                
                if st.button("Reset Preprocessing (Restore Raw Data)", key="reset_preproc"):
                    import copy
                    st.session_state.sp_obj = copy.deepcopy(obj_raw)
                    st.rerun()
            else:
                st.info("Run an analysis or apply preprocessing to visualize.")

else:
    # WITec Raman Pipeline GUI
    st.header("WITec Raman Hyperspectral Imaging Pipeline")
    st.markdown("""
    This processing pipeline is designed for WITec Raman map exports (comma-delimited `.txt` format).
    It executes background subtraction, cosmic ray spike filtering, spectral cropping, baseline fitting, 
    normalisation, VCA unmixing, and abundance mapping.
    """)
    
    st.sidebar.header("Pipeline Data Input")
    
    # Scan input
    scan_source = st.sidebar.radio("Scan File Source", ["Upload File", "Local File Path"])
    scan_path = None
    if scan_source == "Upload File":
        uploaded_scan = st.sidebar.file_uploader("Upload Scan File (.txt)", type=["txt"])
        if uploaded_scan is not None:
            scan_path = save_uploaded_file(uploaded_scan, "temp_scan.txt")
    else:
        col_scan_path, col_scan_btn = st.sidebar.columns([3, 1])
        scan_path_input = col_scan_path.text_input("Local Scan File Path (.txt)", value=st.session_state.get("scan_path_input", ""))
        if col_scan_btn.button("Browse...", key="browse_scan"):
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            selected_file = filedialog.askopenfilename(master=root, filetypes=[("Text files", "*.txt")], title="Select Scan File")
            root.destroy()
            if selected_file:
                st.session_state.scan_path_input = selected_file
                st.rerun()
        if scan_path_input:
            scan_path = scan_path_input
            
    # Glass input
    use_glass = st.sidebar.checkbox("Use Glass/Background Subtraction", value=True)
    glass_path = None
    if use_glass:
        glass_source = st.sidebar.radio("Glass File Source", ["Upload File", "Local File Path"])
        if glass_source == "Upload File":
            uploaded_glass = st.sidebar.file_uploader("Upload Glass/Background File (.txt)", type=["txt"])
            if uploaded_glass is not None:
                glass_path = save_uploaded_file(uploaded_glass, "temp_glass.txt")
        else:
            col_glass_path, col_glass_btn = st.sidebar.columns([3, 1])
            glass_path_input = col_glass_path.text_input("Local Glass/Background File Path (.txt)", value=st.session_state.get("glass_path_input", ""))
            if col_glass_btn.button("Browse...", key="browse_glass"):
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)
                selected_file = filedialog.askopenfilename(master=root, filetypes=[("Text files", "*.txt")], title="Select Glass/Background File")
                root.destroy()
                if selected_file:
                    st.session_state.glass_path_input = selected_file
                    st.rerun()
            if glass_path_input:
                glass_path = glass_path_input

    # Laser preset
    st.sidebar.header("Laser & Preset Configuration")
    preset = st.sidebar.selectbox("Laser Preset", ["532 nm (Fingerprint + C-H stretch)", "785 nm (Fingerprint only)", "Custom"])
    
    if preset == "532 nm (Fingerprint + C-H stretch)":
        p_crop_low = 400.0
        p_crop_high = 3300.0
        p_skip_silent = True
        p_glass_method = "vector"
        p_airpls_strength = 1e3
        p_norm_mode = "dual"
    elif preset == "785 nm (Fingerprint only)":
        p_crop_low = 400.0
        p_crop_high = 1950.0
        p_skip_silent = False
        p_glass_method = "lsq"
        p_airpls_strength = 1e5
        p_norm_mode = "single"
    else: # Custom
        p_crop_low = 400.0
        p_crop_high = 3300.0
        p_skip_silent = True
        p_glass_method = "vector"
        p_airpls_strength = 1e3
        p_norm_mode = "dual"

    # Pipeline Analysis Selection
    st.sidebar.header("Analysis Selection")
    pipeline_analysis = st.sidebar.selectbox("Pipeline Analysis Method", ["VCA (Unmixing)", "HCA (Clustering)", "PCA (Principal Components)"])

    # Define defaults to avoid NameErrors
    n_endmembers = 8
    pca_components = 3
    endmember_labels_input = ""
    map_interpolation = "nearest"
    hca_distance = "euclidean"
    hca_linkage = "ward"
    hca_dist = 1.0
    truncate_p_val = None
    truncate_dendrogram = False
    smooth_method = "None"
    smooth_savgol_window = 15
    smooth_savgol_polyorder = 3
    smooth_gaussian_sigma = 2.0
    smooth_spatial_sigma = 0.0

    with st.sidebar.expander("Advanced Pipeline Parameters", expanded=(preset == "Custom")):
        crop_low = st.number_input("Crop Low (cm-1)", value=p_crop_low)
        crop_high = st.number_input("Crop High (cm-1)", value=p_crop_high)
        skip_silent = st.checkbox("Exclude Raman Silent Region (1900-2600 cm-1)", value=p_skip_silent)
        
        glass_methods = ["None", "direct", "vector", "lsq"]
        glass_method = st.selectbox("Glass Subtraction Method", glass_methods, index=glass_methods.index(p_glass_method) if p_glass_method in glass_methods else 0)
        
        norm_modes = ["single", "dual"]
        norm_mode = st.selectbox("Normalisation Mode", norm_modes, index=norm_modes.index(p_norm_mode) if p_norm_mode in norm_modes else 0)
        
        airpls_strength = st.number_input("airPLS Strength (lambda)", value=float(p_airpls_strength), format="%e")
        cosmic_ray_threshold = st.slider("Cosmic Ray Threshold (z-score)", min_value=1.0, max_value=15.0, value=4.5, step=0.5)
        airpls_itermax = st.number_input("airPLS Max Iterations", min_value=10, max_value=200, value=50)
        
        st.markdown("---")
        st.markdown("##### Spectral Smoothing")
        smooth_method = st.selectbox("Smoothing Method", ["None", "savgol", "gaussian"], index=0)
        if smooth_method == "savgol":
            smooth_savgol_window = st.number_input("Savgol Window Size (odd integer)", min_value=3, value=15, step=2)
            if smooth_savgol_window % 2 == 0:
                smooth_savgol_window += 1
            smooth_savgol_polyorder = st.number_input("Savgol Polynomial Order", min_value=1, max_value=smooth_savgol_window-1, value=3, step=1)
        elif smooth_method == "gaussian":
            smooth_gaussian_sigma = st.number_input("Gaussian Sigma (std dev)", min_value=0.1, value=2.0, step=0.1)
            
        st.markdown("---")
        st.markdown("##### Spatial Smoothing")
        smooth_spatial_sigma = st.number_input("Spatial Gaussian Sigma (0.0 to disable)", min_value=0.0, value=0.0, step=0.1, help="Applies a 2D Gaussian filter spatially over the pixels for each channel.")
        
        if pipeline_analysis == "VCA (Unmixing)":
            n_endmembers = st.slider("VCA Endmembers", min_value=2, max_value=20, value=8)
            endmember_labels_input = st.text_input("Endmember Labels (comma-separated, optional)", value="", help="e.g. PET,PMMA,glass")
            map_interpolation = st.selectbox("Abundance Map Interpolation", ["nearest", "bilinear", "none"], index=0)
        elif pipeline_analysis == "PCA (Principal Components)":
            pca_components = st.slider("PCA Components", min_value=1, max_value=20, value=3)
        else: # HCA (Clustering)
            hca_distance = st.sidebar.selectbox("HCA Distance Metric", ["euclidean", "cosine", "manhattan", "pearson"])
            if hca_distance in ["cosine", "manhattan"]:
                hca_linkage = st.sidebar.selectbox("HCA Linkage Method", ["complete", "average", "single"])
            else:
                hca_linkage = st.sidebar.selectbox("HCA Linkage Method", ["ward", "complete", "average", "single"])
            hca_dist = st.sidebar.number_input("HCA Distance Threshold (dist)", min_value=0.0, value=1.0, step=0.1, help="Cut-off threshold for dendrogram branches")
            truncate_dendrogram = st.sidebar.checkbox("Truncate Dendrogram View", value=False)
            if truncate_dendrogram:
                truncate_p_val = st.sidebar.number_input("Number of Branches (p)", min_value=2, value=10, step=1)
        
        st.markdown("---")
        st.markdown("##### Optics / Acquisition (Optional)")
        laser_wavelength = st.text_input("Laser Wavelength", value="", help="e.g. 532 nm")
        integration_time = st.number_input("Integration Time (s)", min_value=0.0, value=0.0, step=0.1)
        laser_power = st.number_input("Laser Power (mW)", min_value=0.0, value=0.0, step=1.0)
        objective = st.text_input("Objective", value="", help="e.g. 100x / 0.9 NA")
        grating = st.text_input("Grating", value="", help="e.g. 600 g/mm")
        accumulations = st.number_input("Accumulations", min_value=0, value=0, step=1)
        
        st.markdown("---")
        col_out_path, col_out_btn = st.columns([3, 1])
        custom_output_dir = col_out_path.text_input("Output Directory (Optional)", value=st.session_state.get("custom_output_dir", ""))
        if col_out_btn.button("Browse...", key="browse_outdir"):
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            selected_dir = filedialog.askdirectory(master=root, title="Select Output Directory")
            root.destroy()
            if selected_dir:
                st.session_state.custom_output_dir = selected_dir
                st.rerun()

    col_btn1, col_btn2 = st.sidebar.columns(2)
    run_pipeline = col_btn1.button("Run Pipeline", type="primary", use_container_width=True)
    save_data_btn = col_btn2.button("Save Data", use_container_width=True)

    if save_data_btn:
        if not st.session_state.get("witec_run_success", False):
            st.sidebar.error("No data to save. Please run the pipeline first.")
        else:
            import shutil
            src_dir = st.session_state.get("witec_out_root")
            dest_dir = custom_output_dir if custom_output_dir else src_dir
            
            if not dest_dir:
                st.sidebar.error("Please specify a valid output directory.")
            elif src_dir == dest_dir:
                st.sidebar.info(f"Data is already saved in: {src_dir}")
            else:
                try:
                    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                    st.sidebar.success(f"Saved copy to: {dest_dir}")
                    # Update session state output path
                    st.session_state.witec_out_root = str(dest_dir)
                except Exception as e:
                    st.sidebar.error(f"Failed to save data: {e}")

    if run_pipeline:
        if not scan_path:
            st.error("Please provide a valid scan file (upload it or enter local path).")
        elif not os.path.exists(scan_path):
            st.error(f"Scan file path does not exist: {scan_path}")
        else:
            # Check glass path and downgrade to warning if missing/invalid
            glass_warning = None
            run_use_glass = use_glass
            if use_glass:
                if not glass_path:
                    glass_warning = "Glass background file not provided. Proceeding without glass subtraction."
                    run_use_glass = False
                elif not os.path.exists(glass_path):
                    glass_warning = f"Glass background file not found: '{glass_path}'. Proceeding without glass subtraction."
                    run_use_glass = False
                    
            if glass_warning:
                st.warning(glass_warning)
                
            # Run pipeline
            log_area = st.empty()
            logs = []
            
            def log(msg):
                logs.append(msg)
                log_area.text_area("Pipeline Console Output", value="\n".join(logs), height=250)
            
            try:
                log("--- WITec Raman Pipeline Started ---")
                log(f"Scan File: {scan_path}")
                if run_use_glass:
                    log(f"Glass File: {glass_path} (Method: {glass_method})")
                else:
                    log("Glass Background Subtraction skipped.")
                
                # Step 1: Load scan
                log("Step 1: Loading scan file...")
                wavenumber, matrix, ncols, nrows = wrp.load_witec_map(scan_path)
                log(f"  Loaded scan: {nrows}x{ncols} pixels, {len(wavenumber)} channels")
                
                # Setup configuration for metadata helper
                cfg_meta = {
                    "SCAN_FILE": scan_path,
                    "GLASS_FILE": glass_path if run_use_glass else None,
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
                
                # Step 2: Cosmic ray removal
                log("Step 2: Removing cosmic rays...")
                matrix, n_fixed = wrp.remove_cosmic_rays(matrix, nrows, ncols, threshold=cosmic_ray_threshold)
                log(f"  Cosmic ray removal complete. {n_fixed} spikes fixed.")
                
                # Step 3: Glass subtraction
                if run_use_glass and glass_path and glass_method != "None":
                    log("Step 3: Loading glass spectrum and performing subtraction...")
                    glass_wn, glass_int = wrp.load_spectrum(glass_path)
                    glass_interp = np.interp(wavenumber, glass_wn, glass_int)
                    
                    if glass_method == "direct":
                        matrix = wrp.subtract_glass_direct(matrix, glass_interp)
                    elif glass_method == "vector":
                        matrix = wrp.subtract_glass_vector(matrix, glass_interp)
                    elif glass_method == "lsq":
                        matrix = wrp.subtract_glass_lsq(matrix, glass_interp)
                    log(f"  Glass subtraction complete using '{glass_method}' method.")
                else:
                    log("Step 3: Glass subtraction skipped.")
                    glass_wn, glass_int = None, None
                
                # Step 4: Spatial Gaussian smoothing
                if smooth_spatial_sigma > 0.0:
                    log("Step 4: Applying spatial Gaussian smoothing...")
                    matrix = wrp.spatial_gaussian_smooth(matrix, nrows, ncols, smooth_spatial_sigma)
                    log(f"  Spatial Gaussian smoothing complete (sigma={smooth_spatial_sigma:.1f}).")
                else:
                    log("Step 4: Spatial Gaussian smoothing skipped.")

                # Step 5: Spectral smoothing
                if smooth_method and smooth_method != "None":
                    log("Step 5: Applying spectral smoothing...")
                    matrix = wrp.smooth_spectra(
                        matrix,
                        smooth_method,
                        window=smooth_savgol_window,
                        polyorder=smooth_savgol_polyorder,
                        sigma=smooth_gaussian_sigma
                    )
                    log(f"  Spectral smoothing complete ({smooth_method}).")
                else:
                    log("Step 5: Spectral smoothing skipped.")
                
                # Step 6: Crop
                log("Step 6: Cropping spectra...")
                wavenumber, matrix = wrp.crop_spectrum(wavenumber, matrix, low=crop_low, high=crop_high, skip_silent=skip_silent)
                log(f"  Cropped to {len(wavenumber)} channels ({wavenumber.min():.0f}-{wavenumber.max():.0f} cm-1)")
                
                # Step 7: Baseline correction
                log("Step 7: Applying airPLS baseline correction (this may take a moment)...")
                matrix = wrp.correct_baseline(matrix, lam=airpls_strength, itermax=airpls_itermax)
                log("  Baseline correction complete.")
                
                # Step 8: Normalisation
                log("Step 8: Normalising spectra...")
                matrix = wrp.normalise(matrix, wavenumber, mode=norm_mode)
                log(f"  Normalisation complete ({norm_mode}).")
                
                # Setup output folder path
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = Path(scan_path).stem
                if custom_output_dir:
                    out_root = Path(custom_output_dir)
                else:
                    out_root = Path(scan_path).parent / f"{stem}_{timestamp}"
                fig_dir = out_root / "figures"
                proc_dir = out_root / "processed"
                fig_dir.mkdir(parents=True, exist_ok=True)
                proc_dir.mkdir(parents=True, exist_ok=True)
                
                if pipeline_analysis == "VCA (Unmixing)":
                    # Step 9: VCA unmixing
                    log("Step 9: Running VCA unmixing...")
                    Ae = wrp.vca(matrix, n_endmembers)
                    log(f"  VCA unmixing complete. {n_endmembers} endmembers identified.")
                    
                    # Step 10: NNLS abundances
                    log("Step 10: Computing NNLS abundance maps...")
                    abundances = wrp.compute_abundances(matrix, Ae, nrows, ncols)
                    log("  Abundance maps computed successfully.")
                    
                    # Save CSVs
                    endmember_path = str(proc_dir / "endmember_spectra.csv")
                    abundance_path = str(proc_dir / "abundance_maps.csv")
                    labels = [lbl.strip() for lbl in endmember_labels_input.split(",")] if endmember_labels_input else None
                    wrp.export_endmembers(Ae, wavenumber, endmember_path, labels=labels, header_str=header_str)
                    wrp.export_abundances(abundances, nrows, ncols, abundance_path, labels=labels, header_str=header_str)
                    
                    # Save Figures
                    if run_use_glass and glass_path and glass_method != "None":
                        wrp.plot_glass(glass_wn, glass_int, str(fig_dir / "glass_spectrum.png"), 150, skip_silent)
                    wrp.plot_endmembers(Ae, wavenumber, skip_silent, str(fig_dir / "vca_endmembers.png"), 150, labels=labels)
                    wrp.plot_abundance_maps(abundances, str(fig_dir / "abundance_maps.png"), 150, labels=labels, interpolation=map_interpolation)
                    
                    st.session_state.witec_abundances = abundances
                    st.session_state.witec_endmembers = Ae
                    st.session_state.witec_wavenumber = wavenumber
                    st.session_state.witec_skip_silent = skip_silent
                    st.session_state.witec_n_endmembers = n_endmembers
                    st.session_state.witec_labels = labels
                    st.session_state.witec_map_interpolation = map_interpolation
                    
                    # Load previews
                    st.session_state.witec_df_endmembers = pd.read_csv(endmember_path, comment="#")
                    st.session_state.witec_df_abundances = pd.read_csv(abundance_path, comment="#")
                elif pipeline_analysis == "PCA (Principal Components)":
                    log("Step 9: Running PCA on spectral matrix...")
                    scores, loadings, variance_ratio = wrp.run_pca(matrix, pca_components)
                    log(f"  PCA complete. {pca_components} components extracted.")
                    
                    # Construct coordinates for mapping
                    xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
                    pos_df = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten()})
                    
                    # Save PCA scores and loadings to CSV
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
                        
                    # Save figures
                    # a. Score maps
                    for i in range(pca_components):
                        pc_num = i + 1
                        score_vals = scores[i, :]
                        aux = np.zeros((ncols, nrows))
                        aux[:] = np.nan
                        for idx1 in range(matrix.shape[1]):
                            xi = xs.flatten()[idx1]
                            yi = ys.flatten()[idx1]
                            if 0 <= xi < ncols and 0 <= yi < nrows:
                                aux[xi, yi] = score_vals[idx1]
                                
                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(np.rot90(aux, 1, axes=(0, 1)), cmap="coolwarm", interpolation="nearest")
                        ax.set_title(f"PC {pc_num} Score Map", fontsize=11, fontweight="bold")
                        ax.axis("off")
                        plt.colorbar(im, ax=ax)
                        fig.tight_layout()
                        fig.savefig(str(fig_dir / f"pca_score_map_pc{pc_num}.png"), dpi=150)
                        plt.close(fig)
                        
                    # b. Loadings
                    wn_num = pd.to_numeric(wavenumber)
                    for i in range(pca_components):
                        pc_num = i + 1
                        load_vals = loadings[i, :]
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(wn_num, load_vals, color="#2ca02c", lw=1.8)
                        ax.set_title(f"PC {pc_num} Loadings", fontsize=11, fontweight="bold")
                        ax.set_xlabel("Wavenumber (cm-1)")
                        ax.set_ylabel("Loading Weight")
                        ax.set_xlim(wn_num.min(), wn_num.max())
                        ax.grid(ls="--", alpha=0.3)
                        fig.tight_layout()
                        fig.savefig(str(fig_dir / f"pca_loadings_pc{pc_num}.png"), dpi=150)
                        plt.close(fig)
                        
                    # c. PCA scatter
                    fig, ax = plt.subplots(figsize=(6, 5))
                    x_vals = scores[0, :]
                    y_vals = scores[min(1, pca_components-1), :]
                    ax.scatter(x_vals, y_vals, alpha=0.7, color="#1f77b4", edgecolors="none")
                    ax.set_xlabel("PC 1")
                    ax.set_ylabel(f"PC {min(2, pca_components)}")
                    ax.set_title("PCA Score scatter plot", fontsize=11, fontweight="bold")
                    ax.grid(ls="--", alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(str(fig_dir / "pca_scatter.png"), dpi=150)
                    plt.close(fig)
                    
                    if run_use_glass and glass_path and glass_method != "None":
                        wrp.plot_glass(glass_wn, glass_int, str(fig_dir / "glass_spectrum.png"), 150, skip_silent)
                        
                    st.session_state.witec_pca_scores = scores
                    st.session_state.witec_pca_loadings = loadings
                    st.session_state.witec_pca_variance = variance_ratio
                    st.session_state.witec_pca_wavenumber = wavenumber
                    st.session_state.witec_pca_m = ncols
                    st.session_state.witec_pca_n = nrows
                    st.session_state.witec_pca_position = pos_df
                    st.session_state.witec_pca_components = pca_components
                    st.session_state.witec_df_pca_scores = scores_df
                    st.session_state.witec_df_pca_loadings = loadings_df
                else: # HCA Clustering
                    log("Step 9: Instantiating HCA clustering...")
                    hca_obj = sp.hyper_object("hca_witec")
                    hca_obj.data = pd.DataFrame(matrix.T, columns=wavenumber)
                    xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
                    hca_obj.position = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten()})
                    hca_obj.m = ncols
                    hca_obj.n = nrows
                    hca_obj.resolution = 1
                    hca_obj.sublabel = pd.Series(np.zeros(matrix.shape[1]), name="sublabel")
                    hca_obj.label = pd.Series([1]*matrix.shape[1])
                    
                    log("Step 10: Computing HCA dendrogram tree and cluster groupings...")
                    plt.close('all')
                    hca_obj.hca(hca_distance, hca_linkage, hca_dist, truncate_p_val)
                    hca_fig = plt.gcf()
                    hca_fig.savefig(str(fig_dir / "hca_dendrogram.png"), dpi=150, bbox_inches="tight")
                    
                    log("  Computing HCA clustering spatial map...")
                    plt.close('all')
                    colors = hca_obj.show_map('auto', None, 1)
                    map_fig = plt.gcf()
                    map_fig.savefig(str(fig_dir / "hca_cluster_map.png"), dpi=150, bbox_inches="tight")
                    
                    log("  Computing HCA clustered spectra stack...")
                    plt.close('all')
                    hca_obj.show_stack(0.1, 0.5, 'auto')
                    stack_fig = plt.gcf()
                    stack_fig.savefig(str(fig_dir / "hca_spectra_stack.png"), dpi=150, bbox_inches="tight")
                    
                    # Save CSV Data
                    export_df = pd.concat([hca_obj.position, hca_obj.label, hca_obj.data], axis=1)
                    hca_csv_path = str(proc_dir / "hca_clustering.csv")
                    with open(hca_csv_path, "w") as fh:
                        fh.write(header_str)
                        export_df.to_csv(fh, index=False)
                        
                    # Save Glass Background Spectrum if subtraction happened
                    if run_use_glass and glass_path and glass_method != "None":
                        wrp.plot_glass(glass_wn, glass_int, str(fig_dir / "glass_spectrum.png"), 150, skip_silent)
                    
                    st.session_state.witec_df_hca = pd.read_csv(hca_csv_path, comment="#")
                
                # Save full JSON metadata file
                meta_path = proc_dir / "metadata.json"
                with open(meta_path, "w") as fh:
                    json.dump(meta, fh, indent=4)
                
                log(f"  All files successfully saved to: {out_root}")
                log("--- WITec Raman Pipeline Success ---")
                
                # Store in session state for persistence
                st.session_state.witec_run_success = True
                st.session_state.witec_out_root = str(out_root)
                st.session_state.witec_use_glass = run_use_glass and glass_path and glass_method != "None"
                st.session_state.witec_analysis_method = pipeline_analysis
                
                st.success("Pipeline executed successfully!")
                
            except Exception as e:
                log(f"ERROR: {e}")
                st.error(f"Pipeline failed: {e}")

    # Persistence view
    if st.session_state.get("witec_run_success", False):
        out_root = Path(st.session_state.witec_out_root)
        analysis_method = st.session_state.get("witec_analysis_method", "VCA (Unmixing)")
        
        st.markdown("---")
        st.success(f"Pipeline Results (Saved at: `{out_root}`)")
        
        if analysis_method == "VCA (Unmixing)":
            n_endmembers = st.session_state.get("witec_n_endmembers", 8)
            abundances = st.session_state.get("witec_abundances")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Abundance Maps", "VCA Endmembers", "Biochemical Quantification", "Glass Spectrum", "Data Tables"])
            
            with tab1:
                st.subheader("Abundance Maps Grid")
                abundance_img = out_root / "figures" / "abundance_maps.png"
                if abundance_img.exists():
                    st.image(str(abundance_img), use_container_width=True)
                else:
                    st.warning("Abundance maps image not found on disk.")
                    
                st.markdown("---")
                st.subheader("Interactive Component Zoom & Spectrum Viewer")
                
                # Select component using labels if available
                labels = st.session_state.get("witec_labels")
                map_interpolation = st.session_state.get("witec_map_interpolation", "nearest")
                
                comp_options = []
                for idx in range(n_endmembers):
                    lbl = labels[idx] if (labels and idx < len(labels)) else f"Endmember {idx+1}"
                    comp_options.append(lbl)
                    
                selected_comp = st.selectbox("Select Component to Zoom", comp_options)
                comp_idx = comp_options.index(selected_comp)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"##### {selected_comp} Reference Spectrum")
                    Ae = st.session_state.get("witec_endmembers")
                    wavenumber = st.session_state.get("witec_wavenumber")
                    skip_silent = st.session_state.get("witec_skip_silent", True)
                    
                    if Ae is not None and wavenumber is not None:
                        fig_spec, ax_spec = plt.subplots(figsize=(6, 5))
                        wn_d = wrp._display_axis(wavenumber, skip_silent)
                        ax_spec.plot(wn_d, Ae[:, comp_idx], color="#1f77b4", lw=1.8)
                        
                        # Find and label prominent peaks
                        from scipy.signal import savgol_filter, find_peaks
                        window_len = 15
                        if window_len >= len(Ae[:, comp_idx]):
                            window_len = len(Ae[:, comp_idx]) - 1
                            if window_len % 2 == 0:
                                window_len -= 1
                        if window_len >= 3:
                            sm = savgol_filter(Ae[:, comp_idx], window_len, 3)
                        else:
                            sm = Ae[:, comp_idx]
                            
                        pks, _ = find_peaks(sm, prominence=sm.max() * 0.05, distance=20)
                        for p in pks[np.argsort(sm[pks])][-5:]:
                            xp = wn_d[p]
                            ax_spec.text(xp, Ae[p, comp_idx] + 0.02 * Ae[:, comp_idx].max(), f"{wavenumber[p]:.0f}",
                                         color="#1f77b4", fontsize=8, fontweight="bold", ha="center")
                                         
                        if skip_silent:
                            ax_spec.text(2000, 0, "//", fontsize=20, fontweight="bold", ha="center", va="bottom")
                            disp_t, orig_l = wrp._xticks_for_display(skip_silent)
                            ax_spec.set_xticks(disp_t); ax_spec.set_xticklabels(orig_l)
                            
                        ax_spec.set_xlabel("Wavenumber (cm-1)")
                        ax_spec.set_ylabel("Intensity (a.u.)")
                        ax_spec.set_xlim(wn_d.min(), wn_d.max())
                        ax_spec.grid(ls="--", alpha=0.3)
                        st.pyplot(fig_spec)
                        plt.close(fig_spec)
                    else:
                        st.warning("Spectra data not available.")
                        
                with col2:
                    st.markdown(f"##### {selected_comp} Abundance Map")
                    fig_map, ax_map = plt.subplots(figsize=(6, 5))
                    im = ax_map.imshow(abundances[:, :, comp_idx], cmap="inferno", interpolation=map_interpolation)
                    ax_map.axis("off")
                    fig_map.colorbar(im, ax=ax_map)
                    st.pyplot(fig_map)
                    plt.close(fig_map)
                
            with tab2:
                st.subheader("VCA Endmembers")
                endmember_img = out_root / "figures" / "vca_endmembers.png"
                if endmember_img.exists():
                    st.image(str(endmember_img), use_container_width=True)
                else:
                    st.warning("Endmembers image not found on disk.")
                    
                st.markdown("---")
                st.subheader("Overlap Endmembers Plot")
                
                Ae = st.session_state.get("witec_endmembers")
                wavenumber = st.session_state.get("witec_wavenumber")
                abundances = st.session_state.get("witec_abundances")
                labels = st.session_state.get("witec_labels")
                skip_silent = st.session_state.get("witec_skip_silent", True)
                map_interpolation = st.session_state.get("witec_map_interpolation", "nearest")
                
                if Ae is not None and wavenumber is not None:
                    em_options = []
                    for idx in range(n_endmembers):
                        lbl = labels[idx] if (labels and idx < len(labels)) else f"Endmember {idx+1}"
                        em_options.append(lbl)
                        
                    selected_ems = st.multiselect("Select Endmembers to Overlap", em_options, default=em_options, key="witec_overlap_select")
                    
                    col_p1, col_p2 = st.columns(2)
                    show_peaks = col_p1.checkbox("Find and Label Peaks", value=True, key="witec_overlap_show_peaks")
                    peak_prominence = col_p2.slider("Peak Prominence (Fraction of max)", min_value=0.01, max_value=0.30, value=0.05, step=0.01, help="Higher values select only more prominent peaks", key="witec_overlap_prominence")
                    
                    if selected_ems:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        wn_d = wrp._display_axis(wavenumber, skip_silent)
                        
                        from scipy.signal import savgol_filter, find_peaks
                        
                        for name in selected_ems:
                            idx = em_options.index(name)
                            # Normalise to [0, 1] for overlapping comparison
                            spec = Ae[:, idx] / (np.max(Ae[:, idx]) or 1)
                            line, = ax.plot(wn_d, spec, label=name, lw=1.5)
                            
                            if show_peaks:
                                # Smooth using Savitzky-Golay
                                sm = savgol_filter(spec, 15, 3)
                                # Find peaks
                                pks, _ = find_peaks(sm, prominence=sm.max() * peak_prominence, distance=20, width=3)
                                # Label top 5 most prominent peaks
                                for p in pks[np.argsort(sm[pks])][-5:]:
                                    xp = wn_d[p]
                                    ax.text(xp, spec[p] + 0.02, f"{wavenumber[p]:.0f}",
                                            color=line.get_color(), fontsize=8, fontweight="bold", ha="center")
                        
                        if skip_silent:
                            ax.text(2000, 0, "//", fontsize=20, fontweight="bold", ha="center", va="bottom")
                            disp_t, orig_l = wrp._xticks_for_display(skip_silent)
                            ax.set_xticks(disp_t); ax.set_xticklabels(orig_l)
                            
                        ax.set_xlabel("Wavenumber (cm-1)")
                        ax.set_ylabel("Normalised Intensity")
                        ax.set_xlim(wn_d.min(), wn_d.max())
                        ax.set_ylim(-0.05, 1.15) # Breathing room for labels at y=1.0
                        ax.set_title("Overlapped Endmember Spectra (Normalised)", fontsize=12, fontweight="bold")
                        ax.legend(frameon=True)
                        ax.grid(ls="--", alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Respective Abundance Maps Grid
                        st.markdown("---")
                        st.markdown("##### Respective Abundance Maps")
                        if abundances is not None:
                            n_sel = len(selected_ems)
                            cols = st.columns(min(n_sel, 4))
                            for i, name in enumerate(selected_ems):
                                idx = em_options.index(name)
                                col = cols[i % 4]
                                
                                fig_map, ax_map = plt.subplots(figsize=(3, 2.5))
                                im = ax_map.imshow(abundances[:, :, idx], cmap="inferno", interpolation=map_interpolation)
                                ax_map.set_title(name, fontsize=10, fontweight="bold")
                                ax_map.axis("off")
                                plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
                                fig_map.tight_layout()
                                
                                col.pyplot(fig_map)
                                plt.close(fig_map)
                        else:
                            st.warning("Abundance maps data not available.")
                    else:
                        st.info("Select at least one endmember to plot.")
                    
            with tab3:
                st.subheader("Biochemical Quantification & Similarity")
                
                # Fetch endmember spectra, wavenumbers and labels
                Ae = st.session_state.get("witec_endmembers")
                wavenumber = st.session_state.get("witec_wavenumber")
                labels = st.session_state.get("witec_labels")
                
                if Ae is not None and wavenumber is not None:
                    # Construct full labels list
                    em_names = []
                    for idx in range(Ae.shape[1]):
                        lbl = labels[idx] if (labels and idx < len(labels)) else f"Endmember {idx+1}"
                        em_names.append(lbl)
                        
                    # Filter to selected/useful endmembers from Tab 2
                    selected_ems = st.session_state.get("witec_overlap_select")
                    if not selected_ems:
                        selected_ems = em_names.copy()
                        
                    selected_indices = [em_names.index(name) for name in selected_ems]
                    Ae_filtered = Ae[:, selected_indices]
                        
                    col_quant1, col_quant2 = st.columns(2)
                    
                    with col_quant1:
                        st.markdown("##### Endmember Correlation Heatmap")
                        if len(selected_ems) < 2:
                            st.info("Select at least 2 endmembers in the 'VCA Endmembers' tab to display the correlation heatmap.")
                        else:
                            # Pearson correlation between spectra columns
                            corr_matrix = np.corrcoef(Ae_filtered.T)
                            
                            fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
                            im_corr = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
                            
                            ax_corr.set_xticks(np.arange(len(selected_ems)))
                            ax_corr.set_yticks(np.arange(len(selected_ems)))
                            ax_corr.set_xticklabels(selected_ems, rotation=45, ha="right", fontsize=9)
                            ax_corr.set_yticklabels(selected_ems, fontsize=9)
                            
                            # Add numeric labels to each cell
                            for i in range(len(selected_ems)):
                                for j in range(len(selected_ems)):
                                    val = corr_matrix[i, j]
                                    text_color = "white" if abs(val) > 0.4 else "black"
                                    ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", 
                                                 color=text_color, fontweight="bold", fontsize=9)
                                                 
                            ax_corr.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold")
                            fig_corr.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04)
                            fig_corr.tight_layout()
                            st.pyplot(fig_corr)
                            plt.close(fig_corr)
                        
                    with col_quant2:
                        st.markdown("##### Peak Intensity Ratios")
                        
                        # Options for standard macromolecule marker ratios
                        ratio_options = [
                            "Lipid / Protein (I_2850 / I_2930)",
                            "Lipid Ester / Protein Amide I (I_1740 / I_1660)",
                            "Lipid / Protein Fingerprint (I_1440 / I_1660)",
                            "Protein Purity: Phenylalanine / Amide I (I_1003 / I_1660)",
                            "DNA / Protein (I_785 / I_1003)",
                            "DNA Phosphate / Protein (I_1095 / I_1003)",
                            "Carbohydrate / Protein (I_1045 / I_1003)",
                            "Lipid Unsaturation (I_1655 / I_1440)",
                            "Custom Ratio"
                        ]
                        selected_ratio = st.selectbox("Select Biochemical Ratio", ratio_options)
                        
                        wn_min, wn_max = wavenumber.min(), wavenumber.max()
                        
                        w1, w2 = None, None
                        ratio_label = ""
                        
                        if selected_ratio == "Lipid / Protein (I_2850 / I_2930)":
                            w1, w2 = 2850.0, 2930.0
                            ratio_label = "Lipid/Protein (I_2850 / I_2930)"
                        elif selected_ratio == "Lipid Ester / Protein Amide I (I_1740 / I_1660)":
                            w1, w2 = 1740.0, 1660.0
                            ratio_label = "Ester/Amide I (I_1740 / I_1660)"
                        elif selected_ratio == "Lipid / Protein Fingerprint (I_1440 / I_1660)":
                            w1, w2 = 1440.0, 1660.0
                            ratio_label = "Lipid/Protein (I_1440 / I_1660)"
                        elif selected_ratio == "Protein Purity: Phenylalanine / Amide I (I_1003 / I_1660)":
                            w1, w2 = 1003.0, 1660.0
                            ratio_label = "Phe/Amide I (I_1003 / I_1660)"
                        elif selected_ratio == "DNA / Protein (I_785 / I_1003)":
                            w1, w2 = 785.0, 1003.0
                            ratio_label = "DNA/Protein (I_785 / I_1003)"
                        elif selected_ratio == "DNA Phosphate / Protein (I_1095 / I_1003)":
                            w1, w2 = 1095.0, 1003.0
                            ratio_label = "DNA/Protein (I_1095 / I_1003)"
                        elif selected_ratio == "Carbohydrate / Protein (I_1045 / I_1003)":
                            w1, w2 = 1045.0, 1003.0
                            ratio_label = "Carb/Protein (I_1045 / I_1003)"
                        elif selected_ratio == "Lipid Unsaturation (I_1655 / I_1440)":
                            w1, w2 = 1655.0, 1440.0
                            ratio_label = "Unsaturation (I_1655 / I_1440)"
                        else:
                            st.markdown("**Enter custom wavenumbers:**")
                            col_c1, col_c2 = st.columns(2)
                            w1 = col_c1.number_input("Wavenumber 1 (Numerator)", min_value=float(wn_min), max_value=float(wn_max), value=float(wn_min + (wn_max-wn_min)*0.2), step=1.0)
                            w2 = col_c2.number_input("Wavenumber 2 (Denominator)", min_value=float(wn_min), max_value=float(wn_max), value=float(wn_min + (wn_max-wn_min)*0.8), step=1.0)
                            ratio_label = f"Custom Ratio (I_{w1:.0f} / I_{w2:.0f})"
                            
                        # Validate range
                        if w1 is not None and w2 is not None:
                            in_range1 = wn_min <= w1 <= wn_max
                            in_range2 = wn_min <= w2 <= wn_max
                            
                            if not in_range1 or not in_range2:
                                st.warning(f"Chosen wavenumbers ({w1:.0f} or {w2:.0f} cm-1) are outside "
                                           f"the active range of the cropped spectrum ({wn_min:.0f} to {wn_max:.0f} cm-1).")
                            else:
                                # Find closest indices in the wavenumber array
                                idx1 = np.abs(wavenumber - w1).argmin()
                                idx2 = np.abs(wavenumber - w2).argmin()
                                
                                actual_w1 = wavenumber[idx1]
                                actual_w2 = wavenumber[idx2]
                                
                                st.info(f"Using closest channels: Numerator={actual_w1:.1f} cm-1, Denominator={actual_w2:.1f} cm-1")
                                
                                # Compute ratios for each selected endmember
                                numerators = Ae_filtered[idx1, :]
                                denominators = Ae_filtered[idx2, :]
                                
                                # Avoid division by zero
                                denominators_safe = denominators.copy()
                                denominators_safe[denominators_safe == 0.0] = 1e-10
                                ratio_values = numerators / denominators_safe
                                
                                # Plot ratios
                                fig_ratio, ax_ratio = plt.subplots(figsize=(6, 4.5))
                                bars = ax_ratio.bar(selected_ems, ratio_values, color="#1f77b4", edgecolor="black", alpha=0.85)
                                ax_ratio.set_ylabel("Intensity Ratio Value")
                                ax_ratio.set_title(ratio_label, fontsize=11, fontweight="bold")
                                ax_ratio.set_xticklabels(selected_ems, rotation=45, ha="right", fontsize=9)
                                
                                # Add values on top of bars
                                for bar in bars:
                                    height = bar.get_height()
                                    ax_ratio.annotate(f"{height:.2f}",
                                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                                      xytext=(0, 3),
                                                      textcoords="offset points",
                                                      ha='center', va='bottom', fontsize=8, fontweight="bold")
                                                      
                                ax_ratio.grid(axis="y", ls="--", alpha=0.3)
                                fig_ratio.tight_layout()
                                st.pyplot(fig_ratio)
                                plt.close(fig_ratio)
                else:
                    st.warning("Endmember data is not available yet. Please run the pipeline.")
                    
            with tab4:
                st.subheader("Glass Background Spectrum")
                if st.session_state.witec_use_glass:
                    glass_img = out_root / "figures" / "glass_spectrum.png"
                    if glass_img.exists():
                        st.image(str(glass_img), use_container_width=True)
                    else:
                        st.warning("Glass background spectrum image not found on disk.")
                else:
                    st.info("Glass background subtraction was not applied in this run.")
                    
            with tab5:
                st.subheader("Endmember Spectra Preview (First 50 Rows)")
                st.dataframe(st.session_state.witec_df_endmembers.head(50))
                
                st.subheader("Abundance Maps Preview (First 50 Rows)")
                st.dataframe(st.session_state.witec_df_abundances.head(50))
        elif analysis_method == "PCA (Principal Components)":
            scores = st.session_state.witec_pca_scores
            loadings = st.session_state.witec_pca_loadings
            variance_ratio = st.session_state.witec_pca_variance
            wavenumber = st.session_state.witec_pca_wavenumber
            m = st.session_state.witec_pca_m
            n = st.session_state.witec_pca_n
            pos_df = st.session_state.witec_pca_position
            pca_components = st.session_state.witec_pca_components
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["PCA Score Maps", "PCA Loadings", "PCA Scatter", "Glass Spectrum", "Data Tables"])
            
            with tab1:
                st.subheader("PCA 2D Score Maps Grid")
                cols_per_row = 3
                n_rows = (pca_components + cols_per_row - 1) // cols_per_row
                for row_idx in range(n_rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        pc_idx = row_idx * cols_per_row + col_idx
                        if pc_idx < pca_components:
                            pc_num = pc_idx + 1
                            score_vals = scores[pc_idx, :]
                            
                            aux = np.zeros((m, n))
                            aux[:] = np.nan
                            for idx1 in range(scores.shape[1]):
                                xi = pos_df.iloc[idx1, 0]
                                yi = pos_df.iloc[idx1, 1]
                                if 0 <= xi < m and 0 <= yi < n:
                                    aux[xi, yi] = score_vals[idx1]
                                    
                            fig_map, ax_map = plt.subplots(figsize=(4, 3.5))
                            im = ax_map.imshow(np.rot90(aux, 1, axes=(0, 1)), cmap="coolwarm", interpolation="nearest")
                            ax_map.set_title(f"PC {pc_num} Score Map", fontsize=10, fontweight="bold")
                            ax_map.axis("off")
                            plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
                            fig_map.tight_layout()
                            cols[col_idx].pyplot(fig_map)
                            plt.close(fig_map)
                
            with tab2:
                st.subheader("PCA Loadings Spectrum")
                pc_selection_l = st.selectbox("Select Principal Component for Loadings", [f"PC {i}" for i in range(1, pca_components + 1)], key="witec_pca_load_pc")
                pc_idx_l = int(pc_selection_l.split()[1]) - 1
                
                wn_num = pd.to_numeric(wavenumber)
                load_vals = loadings[pc_idx_l, :]
                
                col_pl1, col_pl2 = st.columns(2)
                show_peaks = col_pl1.checkbox("Find and Label Peaks", value=True, key="witec_pca_loadings_show_peaks")
                peak_prominence = col_pl2.slider("Peak Prominence (Fraction of max)", min_value=0.01, max_value=0.30, value=0.05, step=0.01, key="witec_pca_loadings_prominence")
                
                fig_load, ax_load = plt.subplots(figsize=(10, 4.5))
                ax_load.plot(wn_num, load_vals, color="#2ca02c", lw=1.8)
                
                if show_peaks:
                    from scipy.signal import find_peaks
                    max_val = np.max(np.abs(load_vals)) or 1
                    norm_vals = load_vals / max_val
                    
                    pks_pos, _ = find_peaks(norm_vals, prominence=peak_prominence, distance=20)
                    pks_neg, _ = find_peaks(-norm_vals, prominence=peak_prominence, distance=20)
                    
                    for p in pks_pos:
                        ax_load.text(wn_num[p], load_vals[p] + 0.02 * max_val, f"{wn_num[p]:.0f}",
                                     color="blue", fontsize=8, fontweight="bold", ha="center")
                    for p in pks_neg:
                        ax_load.text(wn_num[p], load_vals[p] - 0.04 * max_val, f"{wn_num[p]:.0f}",
                                     color="red", fontsize=8, fontweight="bold", ha="center")
                                     
                ax_load.set_title(f"{pc_selection_l} Loadings Vector", fontsize=12, fontweight="bold")
                ax_load.set_xlabel("Wavenumber (cm-1)")
                ax_load.set_ylabel("Loading Weight")
                ax_load.set_xlim(wn_num.min(), wn_num.max())
                ax_load.grid(ls="--", alpha=0.3)
                fig_load.tight_layout()
                st.pyplot(fig_load)
                plt.close(fig_load)
                
            with tab3:
                st.subheader("PCA Score Scatter Plot")
                col_sc1, col_sc2 = st.columns(2)
                pc_x = col_sc1.selectbox("X-axis Component", [f"PC {i}" for i in range(1, pca_components + 1)], index=0, key="witec_pca_scat_x")
                pc_y = col_sc2.selectbox("Y-axis Component", [f"PC {i}" for i in range(1, pca_components + 1)], index=min(1, pca_components - 1), key="witec_pca_scat_y")
                
                idx_x = int(pc_x.split()[1]) - 1
                idx_y = int(pc_y.split()[1]) - 1
                
                fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
                x_vals = scores[idx_x, :]
                y_vals = scores[idx_y, :]
                
                ax_sc.scatter(x_vals, y_vals, alpha=0.7, color="#1f77b4", edgecolors="none")
                ax_sc.set_xlabel(pc_x)
                ax_sc.set_ylabel(pc_y)
                ax_sc.set_title(f"PCA Score Projection ({pc_x} vs {pc_y})", fontsize=12, fontweight="bold")
                ax_sc.grid(ls="--", alpha=0.3)
                fig_sc.tight_layout()
                st.pyplot(fig_sc)
                plt.close(fig_sc)
                
            with tab4:
                st.subheader("Glass Background Spectrum")
                if st.session_state.witec_use_glass:
                    glass_img = out_root / "figures" / "glass_spectrum.png"
                    if glass_img.exists():
                        st.image(str(glass_img), use_container_width=True)
                    else:
                        st.warning("Glass background spectrum image not found on disk.")
                else:
                    st.info("Glass background subtraction was not applied in this run.")
                    
            with tab5:
                st.subheader("PCA Processed Data Preview")
                st.markdown("**PCA Scores Preview (First 50 Rows)**")
                st.dataframe(st.session_state.witec_df_pca_scores.head(50))
                st.markdown("**PCA Loadings Preview**")
                st.dataframe(st.session_state.witec_df_pca_loadings)
        else: # HCA (Clustering)
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dendrogram (Tree)", "Cluster Map", "Spectra Stack", "Glass Spectrum", "Data Tables"])
            
            with tab1:
                st.subheader("HCA Dendrogram Tree")
                dendrogram_img = out_root / "figures" / "hca_dendrogram.png"
                if dendrogram_img.exists():
                    st.image(str(dendrogram_img), use_container_width=True)
                else:
                    st.warning("Dendrogram image not found on disk.")
                    
            with tab2:
                st.subheader("HCA Cluster Map (Sections)")
                map_img = out_root / "figures" / "hca_cluster_map.png"
                if map_img.exists():
                    st.image(str(map_img), use_container_width=True)
                else:
                    st.warning("Cluster map image not found on disk.")
                    
            with tab3:
                st.subheader("HCA Clustered Spectra Stack")
                stack_img = out_root / "figures" / "hca_spectra_stack.png"
                if stack_img.exists():
                    st.image(str(stack_img), use_container_width=True)
                else:
                    st.warning("Spectra stack image not found on disk.")
                    
            with tab4:
                st.subheader("Glass Background Spectrum")
                if st.session_state.witec_use_glass:
                    glass_img = out_root / "figures" / "glass_spectrum.png"
                    if glass_img.exists():
                        st.image(str(glass_img), use_container_width=True)
                    else:
                        st.warning("Glass background spectrum image not found on disk.")
                else:
                    st.info("Glass background subtraction was not applied in this run.")
                    
            with tab5:
                st.subheader("HCA Clustering & Spectra Data Preview (First 50 Rows)")
                st.dataframe(st.session_state.witec_df_hca.head(50))
