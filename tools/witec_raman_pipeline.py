#!/usr/bin/env python3
"""
witec_raman_pipeline.py
=======================
General-purpose Witec Raman hyperspectral imaging pipeline.

Supports both 532 nm (fingerprint + C-H stretch) and 785 nm (fingerprint only)
Witec .txt exports. Applies glass subtraction, cosmic-ray removal, airPLS
baseline correction, and VCA unmixing. Saves figures and processed CSVs into
a timestamped output folder.

Quick-start
-----------
1. Fill in SCAN_FILE and GLASS_FILE in the USER CONFIGURATION block below.
2. Pick the preset for your laser line (532 nm or 785 nm) — one comment to change.
3. Run::

       python witec_raman_pipeline.py

Output
------
A timestamped folder is created alongside the scan file::

    <scan_stem>_YYYYMMDD_HHMMSS/
        figures/
            glass_spectrum.png
            vca_endmembers.png
            abundance_maps.png
        processed/
            endmember_spectra.csv
            abundance_maps.csv

Requirements
------------
    numpy, scipy, matplotlib
    joblib  (optional — enables parallel baseline correction)
"""

from __future__ import annotations

import os
import re
import sys
import datetime
import json
from pathlib import Path

import numpy as np
import scipy.linalg as splin
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize as opt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — change to "TkAgg" for pop-up windows
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ←  the only section you need to edit
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Input files ---------------------------------------------------------─
    #   SCAN_FILE  : Witec map export (.txt, comma-delimited).
    #   GLASS_FILE : Background spectrum (.txt). Set to None to skip subtraction.
    "SCAN_FILE":  r"C:\path\to\your\scan.txt",
    "GLASS_FILE": r"C:\path\to\your\background.txt",   # or None

    # ── Laser preset ---------------------------------------------------------
    #   Uncomment ONE of the preset blocks below, or set parameters manually.
    #
    #   532 nm preset (fingerprint + C-H stretch, dual-region normalisation):
    "CROP_LOW":    400,    # cm-1  lower wavenumber bound
    "CROP_HIGH":  3300,    # cm-1  upper wavenumber bound
    "SKIP_SILENT": True,   # exclude 1900-2600 cm-1 (Raman silent region)
    "GLASS_METHOD": "vector",  # 'vector' | 'lsq' | None  (see below)
    "AIRPLS_STRENGTH": 1e3,    # baseline smoothness lambda  (1e3 for 532, 1e5 for 785)
    "NORM_MODE": "dual",       # 'dual' | 'single'
        #
    #   785 nm preset — replace the lines above with these:
    # "CROP_LOW":    400,
    # "CROP_HIGH":  1950,
    # "SKIP_SILENT": False,
    # "GLASS_METHOD": "lsq",
    # "AIRPLS_STRENGTH": 1e5,
    # "NORM_MODE": "single",
    
    # ── Glass subtraction options ---------------------------------------------
    #   'direct' : Direct subtraction (pixel - glass). Best when both were
    #              acquired with identical integration time and accumulations.
    #   'vector' : L2-normalised subtraction — best when dwell times differ.
    #              Normalises pixel and glass to unit norm, subtracts, restores scale.
    #   'lsq'    : Per-pixel least-squares fit (alpha·glass + offset). Robust when
    #              glass contribution varies spatially. alpha is clamped ≥ 0.
    #   None     : No subtraction.

    # ── Normalisation options ------------------------------------------------─
    #   'single' : Scales the entire spectrum to an L2 norm of 1. Perfect for
    #              785 nm data which typically only covers the fingerprint region.
    #   'dual'   : Splits the spectrum, scaling the fingerprint (<= 1900 cm-1) and
    #              C-H stretch (>= 2600 cm-1) independently to 1. Essential for
    #              532 nm biological data so the bright C-H peak doesn't overshadow
    #              subtle fingerprint features during unmixing.

    # ── Cosmic ray removal ---------------------------------------------------─
    #   Lower threshold -> more aggressive removal. Typical range 4-9.
    "COSMIC_RAY_THRESHOLD": 4.5,

    # ── airPLS baseline correction ------------------------------------------──
    "AIRPLS_ITERMAX": 50,    # maximum iterations

    # ── VCA endmember unmixing ------------------------------------------------
    "N_ENDMEMBERS": 8,
    "ENDMEMBER_LABELS": None,  # List of string labels (e.g. ["PET", "PMMA", ...]) or None
    "MAP_INTERPOLATION": "nearest", # 'nearest' | 'bilinear' | 'none' etc. for abundance maps

    # ── Optics / Acquisition parameters (optional) ----------------------------
    "LASER_WAVELENGTH": None,       # e.g. "532 nm"
    "INTEGRATION_TIME_SEC": None,   # e.g. 0.5
    "LASER_POWER_MW": None,         # e.g. 10.0
    "OBJECTIVE": None,              # e.g. "100x / 0.9 NA"
    "GRATING": None,                # e.g. "600 g/mm"
    "ACCUMULATIONS": None,          # e.g. 1

    # ── Spectral smoothing ----------------------------------------------------
    "SMOOTH_METHOD": None,          # 'savgol' | 'gaussian' | None
    "SMOOTH_SAVGOL_WINDOW": 15,     # window length (must be odd)
    "SMOOTH_SAVGOL_POLYORDER": 3,   # polynomial order
    "SMOOTH_GAUSSIAN_SIGMA": 2.0,   # gaussian filter standard deviation

    # ── Spatial smoothing -----------------------------------------------------
    "SPATIAL_GAUSSIAN_SIGMA": 0.0,  # spatial 2D Gaussian filter standard deviation (0.0 to disable)

    # ── Output ---------------------------------------------------------------─
    #   OUTPUT_DIR : explicit output path, or None to auto-create next to scan file.
    "OUTPUT_DIR": None,
    "FIGURE_DPI": 150,
}

# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── I/O ---------------------------------------------------------------------──

def load_witec_map(path: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load a Witec comma-delimited map export.

    Returns
    -------
    wavenumber : (L,) array of Raman shifts in cm-1.
    matrix     : (L, N) array of intensities, one column per pixel.
    ncols      : number of X pixels (fastest-varying spatial axis).
    nrows      : number of Y pixels.

    Grid dimensions are parsed from the ``(x/y)`` coordinate pairs in the
    column header.  Falls back to a square grid if the header is absent.
    """
    print(f"Loading scan: {Path(path).name}")
    with open(path) as fh:
        header = fh.readline()

    coords = re.findall(r"\((\d+)/(\d+)\)", header)
    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        data = np.loadtxt(path, delimiter=",", skiprows=1)

    wavenumber = data[:, 0]
    matrix = data[:, 1:]
    n_pixels = matrix.shape[1]

    if coords:
        ncols = max(int(x) for x, _ in coords) + 1
        nrows = max(int(y) for _, y in coords) + 1
        if ncols * nrows != n_pixels:
            raise ValueError(
                f"Header grid {ncols}x{nrows} = {ncols*nrows} "
                f"but data has {n_pixels} pixels."
            )
    else:
        side = int(np.sqrt(n_pixels))
        if side * side != n_pixels:
            raise ValueError(
                f"No (x/y) header found and {n_pixels} pixels is not a perfect square."
            )
        ncols = nrows = side
        print("  No (x/y) header — assuming square grid.")

    print(f"  Grid: {ncols} x {nrows} = {n_pixels} pixels | "
          f"{wavenumber.shape[0]} channels | "
          f"{wavenumber.min():.0f}-{wavenumber.max():.0f} cm-1")
    return wavenumber, matrix, ncols, nrows


def load_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a single-spectrum Witec .txt file (wavenumber, intensity)."""
    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


# ── Spectral crop ------------------------------------------------------------─

def crop_spectrum(wavenumber: np.ndarray, matrix: np.ndarray,
                  low: float, high: float,
                  skip_silent: bool) -> tuple[np.ndarray, np.ndarray]:
    """Crop to [low, high] cm-1, optionally excluding the 1900-2600 silent region."""
    if low >= high:
        raise ValueError(f"Crop low bound ({low}) must be strictly less than crop high bound ({high}).")
    mask = (wavenumber >= low) & (wavenumber <= high)
    if skip_silent:
        mask &= ~((wavenumber > 1900) & (wavenumber < 2600))
    if not np.any(mask):
        raise ValueError(f"Wavenumber crop range [{low}, {high}] resulted in 0 valid spectral channels.")
    wn = wavenumber[mask]
    mat = matrix[mask, :]
    print(f"  Cropped to {wn.shape[0]} channels "
          f"({wn.min():.0f}-{wn.max():.0f} cm-1"
          + (" | silent 1900-2600 excluded" if skip_silent else "") + ")")
    return wn, mat


# ── Glass subtraction ---------------------------------------------------------

def subtract_glass_direct(matrix: np.ndarray,
                           glass: np.ndarray) -> np.ndarray:
    """Direct subtraction (pixel - glass) without scaling.

    Assumes identical exposure time and instrument parameters.
    """
    return matrix - glass[:, np.newaxis]

def subtract_glass_vector(matrix: np.ndarray,
                           glass: np.ndarray) -> np.ndarray:
    """L2 vector-normalised glass subtraction.

    Both the pixel spectra and the glass reference are scaled to unit norm
    before subtraction, making the result independent of integration time.
    The pixel's original scale is then restored.

    Best suited for 532 nm datasets where dwell times may differ between
    map and background acquisition.
    """
    norm_pix = np.linalg.norm(matrix, axis=0)
    norm_pix[norm_pix == 0] = 1.0
    glass_unit = glass / (np.linalg.norm(glass) or 1.0)
    data_unit = matrix / norm_pix
    return (data_unit - glass_unit[:, np.newaxis]) * norm_pix


def subtract_glass_lsq(matrix: np.ndarray,
                        glass: np.ndarray) -> np.ndarray:
    """Per-pixel least-squares glass subtraction.

    Fits ``pixel ≈ alpha·glass + c`` for each pixel and subtracts ``alpha·glass``
    with alpha clamped to ≥ 0 (no negative glass contribution).

    Best suited for 785 nm datasets where glass content varies spatially.
    """
    if np.all(glass == 0) or np.std(glass) < 1e-12:
        print("  Warning: Glass reference is flat or singular; skipping glass subtraction.")
        return matrix
    A = np.column_stack([glass, np.ones_like(glass)])
    try:
        coef, _, rank, _ = np.linalg.lstsq(A, matrix, rcond=None)
        if rank < 2:
            print("  Warning: Glass reference design matrix is singular; skipping glass subtraction.")
            return matrix
        alpha = np.clip(coef[0], 0.0, None)
        print(f"  LSQ glass alpha: min={alpha.min():.3f}  "
              f"median={np.median(alpha):.3f}  max={alpha.max():.3f}")
        return matrix - glass[:, np.newaxis] * alpha[np.newaxis, :]
    except Exception as e:
        print(f"  Warning: Glass subtraction failed ({e}); skipping glass subtraction.")
        return matrix


# ── Cosmic ray removal ------------------------------------------------------──

def remove_cosmic_rays(matrix: np.ndarray, nrows: int, ncols: int,
                        threshold: float = 4.5) -> tuple[np.ndarray, int]:
    """Spatial-spectral consensus cosmic-ray removal.

    Spikes are detected when a channel is simultaneously an outlier in both
    the spectral direction (5-point median) and the spatial direction (3x3
    median), quantified by a robust MAD z-score.  Detected spikes are
    replaced by the spectral median value.

    Parameters
    ----------
    threshold : float
        MAD z-score threshold.  Lower values are more aggressive.
        Typical: 4.5 (sensitive) to 9 (conservative).
    """
    n_ch = matrix.shape[0]
    cube = matrix.reshape((n_ch, nrows, ncols))

    spec_med = median_filter(cube, size=(5, 1, 1))
    spec_diff = cube - spec_med

    spat_med = median_filter(cube, size=(1, 3, 3))
    spat_diff = cube - spat_med

    mad = np.median(np.abs(spec_diff), axis=0)
    mad[mad == 0] = np.mean(mad[mad > 0]) if np.any(mad > 0) else 1.0

    z = spec_diff / (1.4826 * mad)
    spike_mask = (z > threshold) & (spat_diff > threshold * 0.5 * mad)

    cube_clean = cube.copy()
    cube_clean[spike_mask] = spec_med[spike_mask]
    n_fixed = int(np.sum(spike_mask))
    print(f"  Cosmic rays removed: {n_fixed} spike(s)")
    return cube_clean.reshape(matrix.shape), n_fixed


# ── Baseline correction (airPLS) ---------------------------------------------─

def _whittaker_smooth(x: np.ndarray, w: np.ndarray,
                      lam: float, d: int = 1) -> np.ndarray:
    """Penalised least-squares smoother (Whittaker)."""
    m = len(x)
    E = scipy.sparse.eye(m, format="csc")
    D = E[1:] - E[:-1]
    W = scipy.sparse.diags(w, offsets=0, format="csc")
    A = W + lam * D.T @ D
    b = W @ x
    return scipy.sparse.linalg.spsolve(A, b)


def _airpls_single(x: np.ndarray, lam: float,
                   itermax: int = 50) -> np.ndarray:
    """airPLS baseline for a single spectrum."""
    m = len(x)
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = _whittaker_smooth(x, w, lam)
        d = x - z
        neg = d[d < 0]
        dssn = float(np.abs(neg.sum())) if len(neg) else 1.0
        if dssn < 0.001 * float(np.abs(x).sum()) or i == itermax:
            return z
        w[d >= 0] = 0.0
        w[d < 0] = np.exp(i * np.abs(neg) / dssn)
        if len(neg):
            w[0] = w[-1] = np.exp(i * neg.max() / dssn)
    return z


def smooth_spectra(matrix: np.ndarray, method: str | None, **kwargs) -> np.ndarray:
    """Apply spectral smoothing to each spectrum (column) in the matrix.
    
    method: 'savgol' | 'gaussian' | None
    """
    if not method or method == "None":
        return matrix
        
    if method == "savgol":
        window = kwargs.get("window", 15)
        polyorder = kwargs.get("polyorder", 3)
        n_channels = matrix.shape[0]
        if n_channels < 3:
            return matrix
        if window >= n_channels:
            window = n_channels if n_channels % 2 != 0 else n_channels - 1
        if window < 3:
            window = 3
        if polyorder >= window:
            polyorder = max(1, window - 1)
        print(f"  Smoothing spectra using Savitzky-Golay (window={window}, polyorder={polyorder})")
        # Apply filter along the spectral channels axis (axis 0)
        return savgol_filter(matrix, window, polyorder, axis=0)
    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        sigma = kwargs.get("sigma", 2.0)
        print(f"  Smoothing spectra using Gaussian (sigma={sigma})")
        # Apply filter along the spectral channels axis (axis 0)
        return gaussian_filter1d(matrix, sigma, axis=0)
    return matrix


def spatial_gaussian_smooth(matrix: np.ndarray, nrows: int, ncols: int, sigma: float) -> np.ndarray:
    """Apply a 2D spatial Gaussian filter to each spectral channel.
    
    matrix: (n_channels, nrows * ncols)
    """
    if sigma <= 0.0:
        return matrix
        
    print(f"  Applying 2D spatial Gaussian smoothing (sigma={sigma:.1f})")
    from scipy.ndimage import gaussian_filter
    n_channels = matrix.shape[0]
    smoothed = np.empty_like(matrix)
    for c in range(n_channels):
        img2d = matrix[c, :].reshape((nrows, ncols))
        img2d_smoothed = gaussian_filter(img2d, sigma=sigma, mode='reflect')
        smoothed[c, :] = img2d_smoothed.ravel()
    return smoothed


def correct_baseline(matrix: np.ndarray, lam: float,
                      itermax: int = 50) -> np.ndarray:
    """Apply airPLS baseline correction to every pixel spectrum.

    Uses joblib for parallelism when available; falls back to a sequential
    loop otherwise.
    """
    n_pixels = matrix.shape[1]
    print(f"  airPLS baseline (lambda={lam:.0e}, {n_pixels} pixels"
          + (", parallel)" if _JOBLIB else ", sequential)"))

    if _JOBLIB:
        baselines = Parallel(n_jobs=-1)(
            delayed(_airpls_single)(matrix[:, i], lam, itermax)
            for i in range(n_pixels)
        )
    else:
        baselines = [_airpls_single(matrix[:, i], lam, itermax)
                     for i in range(n_pixels)]

    return matrix - np.array(baselines).T



# ── Normalisation ------------------------------------------------------------─

def normalise(matrix: np.ndarray, wavenumber: np.ndarray,
              mode: str) -> np.ndarray:
    """L2 normalise pixel spectra.

    Parameters
    ----------
    mode : 'dual' or 'single'
        ``'dual'``   — independent L2 per region: fingerprint (≤ 1900 cm-1)
                        and C-H stretch (≥ 2600 cm-1).  Balances the two
                        regions when they have very different intensities.
        ``'single'`` — single L2 over the full cropped range.  Suitable for
                        fingerprint-only datasets (785 nm).
    """
    if mode == "dual":
        for mask in [(wavenumber <= 1900), (wavenumber >= 2600)]:
            if not np.any(mask):
                continue
            norms = np.linalg.norm(matrix[mask, :], axis=0)
            norms[norms == 0] = 1.0
            matrix[mask, :] /= norms
        print("  Dual-region L2 normalisation applied.")
    else:
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0] = 1.0
        matrix /= norms
        print("  Single L2 normalisation applied.")
    return matrix


# ── VCA ---------------------------------------------------------------------──

def vca(Y: np.ndarray, R: int, seed: int = 42) -> np.ndarray:
    """Vertex Component Analysis (VCA).

    Finds R endmember spectra that span the data simplex.

    Parameters
    ----------
    Y : (L, N) array — L channels, N pixels.
    R : int          — number of endmembers.
    seed : int       — random seed for reproducibility.

    Returns
    -------
    Ae : (L, R) array of endmember spectra.
    """
    np.random.seed(seed)
    L, N = Y.shape
    max_rank = min(L, N)
    if max_rank == 0:
        raise ValueError("Cannot run VCA on empty dataset.")
    R = int(R)
    if R > max_rank:
        print(f"  Warning: Requested VCA endmembers ({R}) exceeds dataset rank limit ({max_rank}). Clamping to {max_rank}.")
        R = max_rank

    y_m = Y.mean(axis=1, keepdims=True)
    Y_o = Y - y_m
    Ud = splin.svd(Y_o @ Y_o.T / N)[0][:, :R]
    x_p = Ud.T @ Y_o

    P_y = (Y ** 2).sum() / N
    P_x = (x_p ** 2).sum() / N + (y_m ** 2).sum()
    denom = abs(P_y - P_x)
    if denom < 1e-12:
        SNR = 0.0
    else:
        SNR = 10 * np.log10(max(1e-12, abs((P_x - R / L * P_y) / denom)))
    SNR_th = 15 + 10 * np.log10(R)
    print(f"  VCA SNR = {SNR:.1f} dB  (threshold {SNR_th:.1f} dB)")

    if SNR < SNR_th:
        d = R - 1
        Ud = splin.svd(Y_o @ Y_o.T / N)[0][:, :d]
        x_p = Ud.T @ Y_o
        c = float(np.sqrt(np.max((x_p ** 2).sum(axis=0))))
        y = np.vstack([x_p, c * np.ones((1, N))])
    else:
        d = R
        Ud = splin.svd(Y @ Y.T / N)[0][:, :d]
        x_p = Ud.T @ Y
        u = x_p.mean(axis=1, keepdims=True)
        y = x_p / (u.T @ x_p)

    A = np.zeros((R, R))
    A[-1, 0] = 1.0
    indices = np.zeros(R, dtype=int)
    for i in range(R):
        w = np.random.rand(R, 1)
        f = w - A @ (np.linalg.pinv(A) @ w)
        f /= np.linalg.norm(f)
        v = f.T @ y
        indices[i] = int(np.argmax(np.abs(v)))
        A[:, i] = y[:, indices[i]]

    if SNR < SNR_th:
        Ae = Ud @ x_p[:, indices] + y_m
    else:
        Ae = Ud @ x_p[:, indices]

    print(f"  VCA complete - {R} endmembers identified.")
    return Ae


def run_pca(matrix: np.ndarray, n_components: int) -> tuple:
    """Run PCA on a matrix of shape (L, N) -- L channels, N pixels.
    Returns:
        scores: (n_components, N) array
        loadings: (n_components, L) array
        variance_ratio: list of variance explained by each PC
    """
    from sklearn.decomposition import PCA
    L, N = matrix.shape
    max_rank = min(L, N)
    if max_rank == 0:
        raise ValueError("Cannot run PCA on empty dataset.")
    if n_components > max_rank:
        print(f"  Warning: Requested PCA components ({n_components}) exceeds rank limit ({max_rank}). Clamping to {max_rank}.")
        n_components = max_rank
    print(f"  Running PCA (n_components={n_components})...")
    # sklearn.decomposition.PCA expects shape (n_samples, n_features), which is (pixels, channels) -> (N, L)
    pca = PCA(n_components=int(n_components))
    X_transformed = pca.fit_transform(matrix.T) # shape (N, n_components)
    loadings = pca.components_ # shape (n_components, L)
    
    print(f"  PCA complete. Explained variance ratio: {pca.explained_variance_ratio_}")
    return X_transformed.T, loadings, pca.explained_variance_ratio_


# ── NNLS abundances ------------------------------------------------------------

def _nnls_worker(MtM, v):
    return opt.nnls(MtM, v)[0]

def compute_abundances(matrix: np.ndarray, Ae: np.ndarray,
                       nrows: int, ncols: int) -> np.ndarray:
    """Non-negative least squares abundance estimation.

    Decomposes each pixel spectrum as a non-negative combination of the
    endmember spectra.  Uses the MtM pre-computation trick for speed.

    Returns
    -------
    abundance_maps : (nrows, ncols, R) float32 array.
    """
    M = matrix.T                      # (N, L)
    U = Ae.T                          # (R, L)
    MtM = U @ U.T                     # (R, R)
    n_pixels = M.shape[0]
    R = U.shape[0]

    print(f"  NNLS abundances ({n_pixels} pixels, {R} endmembers"
          + (", parallel)" if _JOBLIB else ", sequential)"))

    if _JOBLIB:
        rows = Parallel(n_jobs=-1)(
            delayed(_nnls_worker)(MtM, U @ M[i])
            for i in range(n_pixels)
        )
    else:
        rows = [_nnls_worker(MtM, U @ M[i]) for i in range(n_pixels)]

    return np.array(rows, dtype=np.float32).reshape(nrows, ncols, R)


def nnls_unmix(matrix: np.ndarray, Ae: np.ndarray) -> np.ndarray:
    """Convenience wrapper for non-negative least squares unmixing returning (n_pixels, R)."""
    n_pixels = matrix.shape[1]
    side = int(np.sqrt(n_pixels))
    if side * side == n_pixels:
        nrows = ncols = side
    else:
        nrows = 1
        ncols = n_pixels
    ab = compute_abundances(matrix, Ae, nrows, ncols)
    return ab.reshape(n_pixels, -1)



# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_metadata(cfg: dict, ncols: int, nrows: int) -> tuple[dict, str]:
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "scan_file": str(Path(cfg["SCAN_FILE"]).resolve()),
        "glass_file": str(Path(cfg["GLASS_FILE"]).resolve()) if cfg.get("GLASS_FILE") else None,
        "grid_info": {
            "ncols": ncols,
            "nrows": nrows,
            "total_pixels": ncols * nrows
        },
        "optics_parameters": {
            "laser_wavelength": cfg.get("LASER_WAVELENGTH"),
            "integration_time_sec": cfg.get("INTEGRATION_TIME_SEC"),
            "laser_power_mw": cfg.get("LASER_POWER_MW"),
            "objective": cfg.get("OBJECTIVE"),
            "grating": cfg.get("GRATING"),
            "accumulations": cfg.get("ACCUMULATIONS")
        },
        "processing_parameters": {
            "crop_low": cfg.get("CROP_LOW"),
            "crop_high": cfg.get("CROP_HIGH"),
            "skip_silent": cfg.get("SKIP_SILENT"),
            "glass_method": cfg.get("GLASS_METHOD") if cfg.get("GLASS_FILE") else None,
            "cosmic_ray_threshold": cfg.get("COSMIC_RAY_THRESHOLD"),
            "airpls_strength": cfg.get("AIRPLS_STRENGTH"),
            "airpls_itermax": cfg.get("AIRPLS_ITERMAX"),
            "norm_mode": cfg.get("NORM_MODE"),
            "n_endmembers": cfg.get("N_ENDMEMBERS"),
            "map_interpolation": cfg.get("MAP_INTERPOLATION")
        }
    }
    
    # Format a clean header string for CSV file
    lines = [
        "SpectraMap / WITec Raman Pipeline Export",
        f"Timestamp: {meta['timestamp']}",
        f"Scan File: {Path(meta['scan_file']).name}",
        f"Grid: {ncols}x{nrows} ({ncols*nrows} pixels)"
    ]
    
    opt_lines = []
    for k, v in meta["optics_parameters"].items():
        if v is not None:
            opt_lines.append(f"  {k}: {v}")
    if opt_lines:
        lines.append("Optics / Acquisition Parameters:")
        lines.extend(opt_lines)
        
    lines.append("Processing Parameters:")
    for k, v in meta["processing_parameters"].items():
        lines.append(f"  {k}: {v}")
        
    # Prepend '#' to each line
    header_str = "\n".join(f"# {line}" for line in lines) + "\n"
    return meta, header_str


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _display_axis(wavenumber: np.ndarray, skip_silent: bool) -> np.ndarray:
    """Return a display wavenumber axis that visually closes the silent gap."""
    wn = wavenumber.copy()
    if skip_silent:
        wn[wavenumber >= 2600] -= (2600 - 2100)   # shift C-H region left by 500
    return wn


def _xticks_for_display(skip_silent: bool):
    """Return (display_tick_positions, original_tick_labels) for split axis."""
    orig = np.array([400, 900, 1400, 1900, 2600, 3100])
    disp = orig.copy()
    if skip_silent:
        disp[orig >= 2600] -= (2600 - 2100)
    return disp, [str(v) for v in orig]


def plot_glass(wn: np.ndarray, intensity: np.ndarray,
               out_path: str, dpi: int, skip_silent: bool) -> None:
    wn_d = _display_axis(wn, skip_silent)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wn_d, intensity, color="#E64A19", lw=1.8)
    ax.set_title("Glass Background Spectrum", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel("Intensity (a.u.)")
    if skip_silent:
        ax.text(2000, intensity.min(), "//", fontsize=20, fontweight="bold", ha="center")
        disp_t, orig_l = _xticks_for_display(skip_silent)
        ax.set_xticks(disp_t); ax.set_xticklabels(orig_l)
    ax.grid(ls="--", alpha=0.4)
    ax.set_xlim(wn_d.min(), wn_d.max())
    _save(fig, out_path, dpi)



def plot_endmembers(Ae: np.ndarray, wavenumber: np.ndarray,
                    skip_silent: bool, out_path: str, dpi: int,
                    labels: list[str] | None = None) -> None:
    wn_d = _display_axis(wavenumber, skip_silent)
    R = Ae.shape[1]
    fig, ax = plt.subplots(figsize=(12, 2 + 1.1 * R))
    offset = 1.2
    for i in range(R):
        spec = Ae[:, i] / (np.max(Ae[:, i]) or 1)
        y = spec + i * offset
        line, = ax.plot(wn_d, y, lw=1.4)
        
        # Use custom label if provided and available
        label_text = labels[i] if (labels and i < len(labels)) else f"EM {i+1}"
        
        ax.text(wn_d.min(), y[0] + offset * 0.05,
                label_text, color=line.get_color(),
                fontweight="bold", fontsize=10, va="bottom")
        sm = savgol_filter(spec, 15, 3)
        pks, _ = find_peaks(sm, prominence=sm.max() * 0.05, distance=20, width=3)
        for p in pks[np.argsort(sm[pks])][-5:]:
            xp = wn_d[p]
            ax.text(xp, y[p] + offset * 0.04,
                    f"{wavenumber[p]:.0f}", color=line.get_color(),
                    fontsize=8, fontweight="bold", ha="center")
    if skip_silent:
        ax.text(2000, 0, "//", fontsize=20, fontweight="bold", ha="center", va="bottom")
        disp_t, orig_l = _xticks_for_display(skip_silent)
        ax.set_xticks(disp_t); ax.set_xticklabels(orig_l)
    ax.set_title("VCA Endmembers (stacked, normalised)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel("Intensity (a.u., offset)")
    ax.grid(axis="x", ls="--", alpha=0.3)
    max_y = (R - 1) * offset + 1.35
    ax.set_ylim(-0.1, max_y)
    ax.set_xlim(wn_d.min(), wn_d.max())
    plt.tight_layout()
    _save(fig, out_path, dpi)


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


def plot_abundance_maps(maps: np.ndarray, out_path: str, dpi: int,
                        labels: list[str] | None = None,
                        interpolation: str = "nearest",
                        rotation: int = 0,
                        flip_h: bool = False,
                        flip_v: bool = False) -> None:
    R = maps.shape[2]
    ncols = min(4, R)
    nrows_fig = (R + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows_fig, ncols,
                             figsize=(ncols * 4, nrows_fig * 3.5))
    axes = np.atleast_1d(axes).flatten()
    for i in range(R):
        m_img = orient_map(maps[:, :, i], rotation=rotation, flip_h=flip_h, flip_v=flip_v)
        im = axes[i].imshow(m_img, cmap="inferno", interpolation=interpolation)
        # Use custom label if provided and available
        label_text = labels[i] if (labels and i < len(labels)) else f"EM {i+1}"
        axes[i].set_title(f"{label_text} Abundance", fontsize=10, fontweight="bold")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    for ax in axes[R:]:
        ax.axis("off")
    plt.tight_layout()
    _save(fig, out_path, dpi)


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    base = Path(path).with_suffix("")
    for fmt in [".png", ".pdf", ".svg"]:
        out_f = base.with_suffix(fmt)
        fig.savefig(out_f, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved figure -> {base.name} (.png, .pdf, .svg)")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_endmembers(Ae: np.ndarray, wavenumber: np.ndarray,
                      out_path: str, labels: list[str] | None = None,
                      header_str: str = "") -> None:
    R = Ae.shape[1]
    if labels:
        cols = []
        for i in range(R):
            lbl = labels[i] if i < len(labels) else f"{i+1}"
            cols.append(f"Endmember_{lbl}")
    else:
        cols = [f"Endmember_{i+1}" for i in range(R)]
    header = header_str + "Wavenumber," + ",".join(cols)
    np.savetxt(out_path, np.column_stack([wavenumber, Ae]),
               delimiter=",", header=header, comments="")
    print(f"  Saved -> {Path(out_path).name}")


def export_abundances(maps: np.ndarray, nrows: int, ncols: int,
                      out_path: str, labels: list[str] | None = None,
                      header_str: str = "") -> None:
    R = maps.shape[2]
    xs, ys = np.meshgrid(np.arange(ncols), np.arange(nrows))
    flat = maps.reshape(-1, R)
    export = np.column_stack([xs.flatten(), ys.flatten(), flat])
    if labels:
        cols = []
        for i in range(R):
            lbl = labels[i] if i < len(labels) else f"EM{i+1}"
            cols.append(f"{lbl}_Abundance")
    else:
        cols = [f"EM{i+1}_Abundance" for i in range(R)]
    header = header_str + "X_pixel,Y_pixel," + ",".join(cols)
    np.savetxt(out_path, export, delimiter=",", header=header, comments="")
    print(f"  Saved -> {Path(out_path).name}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    """Execute the full processing pipeline from CONFIG."""

    scan_path = cfg["SCAN_FILE"]
    glass_path = cfg.get("GLASS_FILE")

    # ── Output directories ---------------------------------------------------─
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(scan_path).stem
    if cfg.get("OUTPUT_DIR"):
        out_root = Path(cfg["OUTPUT_DIR"])
    else:
        out_root = Path(scan_path).parent / f"{stem}_{timestamp}"
    fig_dir  = out_root / "figures"
    proc_dir = out_root / "processed"
    fig_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder: {out_root}\n")

    dpi         = cfg.get("FIGURE_DPI", 150)
    skip_silent = cfg.get("SKIP_SILENT", True)

    # ── 1. Load data ---------------------------------------------------------─
    print("--- Step 1: Load data ---")
    wavenumber, matrix, ncols, nrows = load_witec_map(scan_path)
    meta, header_str = get_metadata(cfg, ncols, nrows)

    # ── 2. Cosmic ray removal ------------------------------------------------─
    print("\n--- Step 2: Cosmic ray removal ---")
    matrix, _ = remove_cosmic_rays(
        matrix, nrows, ncols, threshold=cfg.get("COSMIC_RAY_THRESHOLD", 4.5)
    )

    # ── 3. Glass subtraction ------------------------------------------------──
    glass_method = cfg.get("GLASS_METHOD", "vector")
    if glass_path and glass_method:
        print("\n--- Step 3: Glass subtraction ---")
        glass_wn, glass_int = load_spectrum(glass_path)
        plot_glass(glass_wn, glass_int,
                   str(fig_dir / "glass_spectrum.png"), dpi, skip_silent)
        glass_interp = np.interp(wavenumber, glass_wn, glass_int)
        if glass_method == "direct":
            print("  Method: Direct (pixel - glass)")
            matrix = subtract_glass_direct(matrix, glass_interp)
        elif glass_method == "vector":
            print("  Method: L2 vector-normalised")
            matrix = subtract_glass_vector(matrix, glass_interp)
        elif glass_method == "lsq":
            print("  Method: per-pixel least-squares fit")
            matrix = subtract_glass_lsq(matrix, glass_interp)
        print("  Glass subtraction complete.")
    else:
        print("\n--- Step 3: Glass subtraction - skipped ---")

    # ── 4. Spatial Gaussian smoothing ------------------------------------------
    spatial_sigma = cfg.get("SPATIAL_GAUSSIAN_SIGMA", 0.0)
    if spatial_sigma > 0.0:
        print("\n--- Step 4: Spatial Gaussian smoothing ---")
        matrix = spatial_gaussian_smooth(matrix, nrows, ncols, spatial_sigma)
    else:
        print("\n--- Step 4: Spatial Gaussian smoothing - skipped ---")

    # ── 5. Spectral smoothing -------------------------------------------------
    smooth_method = cfg.get("SMOOTH_METHOD")
    if smooth_method and smooth_method != "None":
        print("\n--- Step 5: Spectral smoothing ---")
        matrix = smooth_spectra(
            matrix,
            smooth_method,
            window=cfg.get("SMOOTH_SAVGOL_WINDOW", 15),
            polyorder=cfg.get("SMOOTH_SAVGOL_POLYORDER", 3),
            sigma=cfg.get("SMOOTH_GAUSSIAN_SIGMA", 2.0)
        )
    else:
        print("\n--- Step 5: Spectral smoothing - skipped ---")

    # ── 6. Spectral crop ------------------------------------------------------
    print("\n--- Step 6: Spectral crop ---")
    wavenumber, matrix = crop_spectrum(
        wavenumber, matrix,
        low=cfg.get("CROP_LOW", 400),
        high=cfg.get("CROP_HIGH", 3300),
        skip_silent=skip_silent,
    )

    # ── 7. Baseline correction ------------------------------------------------
    print("\n--- Step 7: Baseline correction (airPLS) ---")
    matrix = correct_baseline(
        matrix,
        lam=cfg.get("AIRPLS_STRENGTH", 1e3),
        itermax=cfg.get("AIRPLS_ITERMAX", 50),
    )

    # ── 8. Normalisation ------------------------------------------------------
    print("\n--- Step 8: Normalisation ---")
    matrix = normalise(matrix, wavenumber, mode=cfg.get("NORM_MODE", "dual"))

    # ── 9. VCA ---------------------------------------------------------------─
    print("\n--- Step 9: VCA unmixing ---")
    R = cfg.get("N_ENDMEMBERS", 8)
    Ae = vca(matrix, R)
    labels = cfg.get("ENDMEMBER_LABELS")
    plot_endmembers(Ae, wavenumber, skip_silent,
                    str(fig_dir / "vca_endmembers.png"), dpi, labels=labels)

    # ── 10. NNLS abundances --------------------------------------------------
    print("\n--- Step 10: NNLS abundances ---")
    abundance_maps = compute_abundances(matrix, Ae, nrows, ncols)
    interpolation = cfg.get("MAP_INTERPOLATION", "nearest")
    plot_abundance_maps(abundance_maps,
                        str(fig_dir / "abundance_maps.png"), dpi, labels=labels, interpolation=interpolation)

    # ── 11. Export CSVs & Figures --------------------------------------------─
    print("\n--- Step 11: Export ---")
    
    # Save full JSON metadata file
    meta_path = proc_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=4)
    print(f"  Saved -> {meta_path.name}")
    
    # Save preprocessed spectra matrix
    import pandas as pd
    preprocessed_path = proc_dir / "preprocessed_spectra.csv"
    df_preprocessed = pd.DataFrame(matrix.T, columns=wavenumber)
    with open(preprocessed_path, "w") as fh:
        fh.write(header_str)
        df_preprocessed.to_csv(fh, index=False)
    print(f"  Saved -> {preprocessed_path.name}")

    export_endmembers(Ae, wavenumber, str(proc_dir / "endmember_spectra.csv"), labels=labels, header_str=header_str)
    export_abundances(abundance_maps, nrows, ncols,
                      str(proc_dir / "abundance_maps.csv"), labels=labels, header_str=header_str)

    # Save Correlation Matrix & Heatmap
    em_names = [labels[i] if (labels and i < len(labels)) else f"EM {i+1}" for i in range(R)]
    corr_matrix = np.corrcoef(Ae.T)
    df_corr = pd.DataFrame(corr_matrix, index=em_names, columns=em_names)
    corr_path = proc_dir / "correlation_matrix.csv"
    with open(corr_path, "w") as fh:
        fh.write(header_str)
        df_corr.to_csv(fh)
    print(f"  Saved -> {corr_path.name}")

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
    _save(fig_corr, str(fig_dir / "correlation_heatmap.png"), dpi)

    # Save Biochemical Peak Ratios Table & Chart
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
    print(f"  Saved -> {ratios_path.name}")

    fig_bio, ax_bio = plt.subplots(figsize=(8, 5))
    df_ratios.plot(kind="bar", ax=ax_bio, colormap="tab10", width=0.8)
    ax_bio.set_ylabel("Peak Intensity Ratio", fontsize=10)
    ax_bio.set_title("Biochemical Macromolecular Peak Ratios", fontsize=12, fontweight="bold")
    ax_bio.set_xticklabels(em_names, rotation=45, ha="right", fontsize=9)
    ax_bio.grid(axis="y", ls="--", alpha=0.3)
    ax_bio.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    _save(fig_bio, str(fig_dir / "biochemical_ratios.png"), dpi)

    print(f"\nDONE  Pipeline complete.  Results in: {out_root}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="General-purpose Witec Raman hyperspectral imaging pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("scan", nargs="?", default=CONFIG["SCAN_FILE"],
                        help="Path to the Witec scan .txt file")
    parser.add_argument("--glass", type=str, default=CONFIG["GLASS_FILE"],
                        help="Path to the glass background .txt file")
    parser.add_argument("--preset", choices=["532", "785", "custom"], default="custom",
                        help="Automatically apply recommended settings for 532nm or 785nm lasers.\n"
                             "If 'custom' is selected, the CONFIG block inside the script is used.")
    parser.add_argument("-o", "--out", "--outdir", dest="outdir", type=str, default=CONFIG["OUTPUT_DIR"],
                        help="Custom output directory (optional)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated list of endmember labels (e.g. 'PET,PMMA,glass')")
    parser.add_argument("--interpolation", "--interp", dest="interpolation", type=str, default=CONFIG["MAP_INTERPOLATION"],
                        help="Interpolation method for abundance maps (e.g., nearest, bilinear, none)")

    args = parser.parse_args()

    # Create a runtime config by copying the script's CONFIG block
    runtime_config = CONFIG.copy()
    
    if args.scan:
        runtime_config["SCAN_FILE"] = args.scan
    if args.glass:
        runtime_config["GLASS_FILE"] = args.glass
    if args.outdir:
        runtime_config["OUTPUT_DIR"] = args.outdir
    if args.labels:
        runtime_config["ENDMEMBER_LABELS"] = [lbl.strip() for lbl in args.labels.split(",")]
    if args.interpolation:
        runtime_config["MAP_INTERPOLATION"] = args.interpolation

    # Apply presets if requested
    if args.preset == "532":
        runtime_config.update({
            "CROP_LOW": 400, "CROP_HIGH": 3300, "SKIP_SILENT": True,
            "GLASS_METHOD": "vector", "AIRPLS_STRENGTH": 1e3, "NORM_MODE": "dual"
        })
    elif args.preset == "785":
        runtime_config.update({
            "CROP_LOW": 400, "CROP_HIGH": 1950, "SKIP_SILENT": False,
            "GLASS_METHOD": "lsq", "AIRPLS_STRENGTH": 1e5, "NORM_MODE": "single"
        })

    if not runtime_config["SCAN_FILE"] or not Path(runtime_config["SCAN_FILE"]).exists():
        print(f"Error: Scan file not found -> {runtime_config['SCAN_FILE']}")
        print("Please provide a valid file via CLI: python witec_raman_pipeline.py <scan_file>")
        sys.exit(1)

    run(runtime_config)
