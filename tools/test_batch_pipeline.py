import sys
from pathlib import Path
import pytest
import numpy as np

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from witec_raman_pipeline import resample_wavenumbers, combine_hyperspectral_datasets

def test_resample_wavenumbers():
    wn_src = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
    mat_src = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ])
    target_wn = np.array([450.0, 650.0])
    
    resampled = resample_wavenumbers(wn_src, mat_src, target_wn)
    assert resampled.shape == (2, 2)
    assert np.isclose(resampled[0, 0], 2.0) # 450 cm-1 interpolated between 1.0 and 3.0
    assert np.isclose(resampled[1, 1], 7.0) # 650 cm-1 interpolated between 6.0 and 8.0

def test_combine_hyperspectral_datasets():
    d1 = {
        'wavenumber': np.linspace(400, 1800, 100),
        'matrix': np.random.rand(100, 25),
        'ncols': 5,
        'nrows': 5,
        'name': 'Sample_1',
        'group': 'Type_A'
    }
    d2 = {
        'wavenumber': np.linspace(450, 1900, 120),
        'matrix': np.random.rand(120, 25),
        'ncols': 5,
        'nrows': 5,
        'name': 'Sample_2',
        'group': 'Type_B'
    }
    
    master = combine_hyperspectral_datasets([d1, d2])
    assert master['is_batch'] is True
    assert master['matrix'].shape[1] == 50 # 25 + 25 pixels
    assert len(master['sample_names']) == 50
    assert len(master['sample_groups']) == 50
    assert set(master['sample_names']) == {'Sample_1', 'Sample_2'}
    assert set(master['sample_groups']) == {'Type_A', 'Type_B'}
