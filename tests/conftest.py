import pytest
import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root, src, and tools to python path
project_root = Path(__file__).parent.parent.resolve()
src_dir = project_root / "src"
tools_dir = project_root / "tools"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

@pytest.fixture
def synthetic_hyperspectral_matrix():
    """Generates a synthetic 10x10 hyperspectral matrix (100 pixels, 300 channels)."""
    np.random.seed(42)
    n_pixels = 100
    n_channels = 300
    wavenumbers = np.linspace(400, 3200, n_channels)
    
    # Define 3 pure Gaussian endmember profiles
    em1 = np.exp(-((wavenumbers - 1000) ** 2) / (2 * 30 ** 2)) + 0.5 * np.exp(-((wavenumbers - 2900) ** 2) / (2 * 40 ** 2))
    em2 = np.exp(-((wavenumbers - 1450) ** 2) / (2 * 25 ** 2)) + 0.8 * np.exp(-((wavenumbers - 2850) ** 2) / (2 * 35 ** 2))
    em3 = np.exp(-((wavenumbers - 1600) ** 2) / (2 * 20 ** 2))
    
    endmembers = np.vstack([em1, em2, em3])  # 3 x 300
    
    # Random Dirichlet-like abundance matrix (100 x 3)
    abundances = np.random.dirichlet((1, 1, 1), size=n_pixels)
    
    # Pure data matrix (100 x 300)
    data = abundances @ endmembers
    
    # Add baseline fluorescence and noise
    baseline = 10.0 + 0.005 * (wavenumbers - 400)
    noise = np.random.normal(0, 0.02, size=(n_pixels, n_channels))
    
    matrix = data + baseline + noise
    matrix = np.clip(matrix, 0, None)
    
    df = pd.DataFrame(matrix, columns=np.round(wavenumbers, 2))
    positions = pd.DataFrame({
        'x': np.repeat(np.arange(10), 10),
        'y': np.tile(np.arange(10), 10)
    })
    
    return {
        'wavenumbers': wavenumbers,
        'endmembers': endmembers,
        'abundances': abundances,
        'df': df,
        'positions': positions,
        'ncols': 10,
        'nrows': 10
    }

@pytest.fixture
def witec_scan_txt_file(tmp_path, synthetic_hyperspectral_matrix):
    """Creates a temporary WITec export format text file."""
    data = synthetic_hyperspectral_matrix
    file_path = tmp_path / "synthetic_witec_scan.txt"
    
    # Build header with (x/y) coordinate pairs
    headers = ["Wavenumber"]
    for y in range(10):
        for x in range(10):
            headers.append(f"({x}/{y})")
    
    with open(file_path, "w") as f:
        f.write(",".join(headers) + "\n")
        matrix = data['df'].values  # 100 x 300
        wavenumbers = data['wavenumbers']
        for i in range(len(wavenumbers)):
            row_str = f"{wavenumbers[i]:.2f}," + ",".join(f"{val:.4f}" for val in matrix[:, i])
            f.write(row_str + "\n")
            
    return str(file_path)

@pytest.fixture
def glass_ref_txt_file(tmp_path, synthetic_hyperspectral_matrix):
    """Creates a temporary glass reference spectrum file."""
    wavenumbers = synthetic_hyperspectral_matrix['wavenumbers']
    file_path = tmp_path / "glass_ref.txt"
    
    # Broad glass fluorescence curve
    glass_intensity = 5.0 + 15.0 * np.exp(-((wavenumbers - 1200) ** 2) / (2 * 400 ** 2))
    
    with open(file_path, "w") as f:
        f.write("Wavenumber,Background\n")
        for wn, val in zip(wavenumbers, glass_intensity):
            f.write(f"{wn:.2f},{val:.4f}\n")
            
    return str(file_path)

@pytest.fixture
def malformed_txt_file(tmp_path):
    """Creates a malformed TXT file with invalid header and non-numeric values."""
    file_path = tmp_path / "malformed.txt"
    with open(file_path, "w") as f:
        f.write("INVALID HEADER WITHOUT COORDS\n")
        f.write("400.0, 1.2, INVALID_NUMBER, 3.4\n")
        f.write("401.0, 1.5, 2.3, 3.8\n")
    return str(file_path)

@pytest.fixture
def nan_inf_matrix_df():
    """Returns a DataFrame containing NaN and Inf values."""
    np.random.seed(42)
    arr = np.random.rand(20, 50)
    arr[2, 5] = np.nan
    arr[10, 15] = np.inf
    arr[12, 20] = -np.inf
    wn = np.linspace(400, 1800, 50)
    return pd.DataFrame(arr, columns=np.round(wn, 2))

@pytest.fixture
def temp_output_dir(tmp_path):
    """Returns a temporary output directory Path."""
    out = tmp_path / "test_output"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)
