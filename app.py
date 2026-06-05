import streamlit as st
import matplotlib.pyplot as plt
import os
import glob
from spectramap import spmap as sp
from smart_importer import parse_with_ollama
import pandas as pd
import numpy as np

st.set_page_config(page_title="SpectraMap GUI", layout="wide")

st.title("SpectraMap GUI")

# Helper function to get data files
def get_data_files():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(data_dir):
        files = glob.glob(os.path.join(data_dir, '*.csv.xz')) + glob.glob(os.path.join(data_dir, '*.spc'))
        return [os.path.basename(f) for f in files]
    return []

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

    st.sidebar.markdown("---")
    st.sidebar.header("Analysis")
    
    analysis_type = st.sidebar.selectbox("Analysis", ["None", "HDBSCAN", "PCA"])
    
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

    # Main view
    st.header("Visualization")
    
    if analysis_type == "HDBSCAN" and st.session_state.get('colors'):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Map")
            try:
                colors = obj.show_map('auto', None, 1)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Error plotting map: {e}")
            plt.close('all')
            
        with col2:
            st.subheader("Stack")
            try:
                obj.show_stack(0.1, 0.5, 'auto')
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Error plotting stack: {e}")
            plt.close('all')
            
    elif analysis_type == "PCA" and st.session_state.get('scores'):
        st.subheader("PCA Scatter")
        try:
            st.session_state.scores.show_scatter(main_label=15, size=15, colors="auto")
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Error plotting PCA: {e}")
        plt.close('all')
        
    else:
        st.info("Run an analysis or apply preprocessing to visualize.")
