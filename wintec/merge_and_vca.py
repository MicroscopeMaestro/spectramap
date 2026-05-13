#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize as opt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter
import os
from joblib import Parallel, delayed

# --- CORE FUNCTIONS ---

def estimate_snr(Y, r_m, x):
    [L, N] = Y.shape           
    [p, N] = x.shape           
    P_y     = np.sum(Y**2)/float(N)
    P_x     = np.sum(x**2)/float(N) + np.sum(r_m**2)
    snr_est = 10*np.log10( (P_x - p/L*P_y)/(P_y - P_x) )
    return snr_est

def vca(Y, R, verbose=True, snr_input=0):
    """
    Vertex Component Analysis (VCA)
    Ae, indice, Yp = vca(Y, R, verbose=True, snr_input=0)
    """
    [L, N] = Y.shape
    R = int(R)
    
    # 1. Determine SNR and project data
    y_m = np.mean(Y, axis=1, keepdims=True)
    if snr_input == 0:
        Y_o = Y - y_m
        Ud = splin.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :R]
        x_p = np.dot(Ud.T, Y_o)
        
        P_y = np.sum(Y**2) / float(N)
        P_x = np.sum(x_p**2) / float(N) + np.sum(y_m**2)
        SNR = 10 * np.log10(abs((P_x - (R / L) * P_y) / (P_y - P_x)))
        if verbose: print(f"SNR estimated = {SNR:.2f}[dB]")
    else:
        SNR = snr_input
        if verbose: print(f"Input SNR = {SNR:.2f}[dB]")

    SNR_th = 15 + 10 * np.log10(R)
    
    # 2. Case differentiation based on SNR
    if SNR < SNR_th:
        d = R - 1
        Ud = splin.svd(np.dot(Y - y_m, (Y - y_m).T) / float(N))[0][:, :d]
        x_p = np.dot(Ud.T, Y - y_m)
        # Projective step: augment with constant to create simplex
        c = np.amax(np.sum(x_p**2, axis=0))**0.5
        y = np.vstack((x_p, c * np.ones((1, N))))
    else:
        d = R
        Ud = splin.svd(np.dot(Y, Y.T) / float(N))[0][:, :d]
        x_p = np.dot(Ud.T, Y)
        # Projective step: normalize by mean projection
        u_p = np.mean(x_p, axis=1, keepdims=True)
        y = x_p / np.dot(u_p.T, x_p)
        
    # 3. VCA Iterations
    np.random.seed(42) # Set seed for reproducibility
    A = np.zeros((R, R))
    A[-1, 0] = 1
    indice = np.zeros(R, dtype=int)
    
    for i in range(R):
        w = np.random.rand(R, 1)
        f = w - np.dot(A, np.dot(splin.pinv(A), w))
        f = f / splin.norm(f)
        v = np.dot(f.T, y)
        indice[i] = np.argmax(np.absolute(v))
        A[:, i] = y[:, indice[i]]
        
    # 4. Final Reconstruction
    if SNR < SNR_th:
        Ae = np.dot(Ud, x_p[:, indice]) + y_m
    else:
        Ae = np.dot(Ud, x_p[:, indice])
        
    return Ae, indice, None

def WhittakerSmooth(x, w, lambda_, differences=1):
    X = np.matrix(x).flatten()
    m = X.size
    E = scipy.sparse.eye(m, format='csc')
    D = E[1:] - E[:-1] 
    W = scipy.sparse.diags(w, 0, shape=(m, m))
    A = scipy.sparse.csc_matrix(W + (lambda_ * D.T * D))
    B = scipy.sparse.csc_matrix(W * X.T)
    background = scipy.sparse.linalg.spsolve(A, B)
    return np.array(background).flatten()

def airPLS(x, lambda_, porder=1, itermax=50):
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        d_neg = d[d < 0]
        dssn = np.abs(d_neg.sum()) if len(d_neg) > 0 else 1.0
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            break
        w[d >= 0] = 0 
        w[d < 0] = np.exp(i * np.abs(d_neg) / dssn)
        if len(d_neg) > 0:
            w[0] = np.exp(i * d_neg.max() / dssn) 
            w[-1] = w[0]
    return z

def remove_cosmic_rays_smart(matrix, grid_size, threshold=4.5):
    n_channels = matrix.shape[0]
    img_3d = matrix.reshape((n_channels, grid_size, grid_size))
    spec_median = median_filter(img_3d, size=(5, 1, 1))
    spec_diff = img_3d - spec_median
    spat_median = median_filter(img_3d, size=(1, 3, 3))
    spat_diff = img_3d - spat_median
    mad = np.median(np.abs(spec_diff), axis=0)
    mad[mad == 0] = np.mean(mad[mad > 0]) if np.any(mad > 0) else 1.0
    z_score_spec = spec_diff / (1.4826 * mad)
    spike_mask = (z_score_spec > threshold) & (spat_diff > (threshold * 0.5 * mad))
    img_clean = img_3d.copy()
    img_clean[spike_mask] = spec_median[spike_mask]
    return img_clean.reshape(matrix.shape), np.sum(spike_mask)

# --- CONFIGURATION ---

data_files = [
    r'C:\Users\Juan\Documents\GitHub\spectramap\wintec\large_CA3.txt',
    r'C:\Users\Juan\Documents\GitHub\spectramap\wintec\large_CA5.txt',
    r'C:\Users\Juan\Documents\GitHub\spectramap\wintec\2D image of CAs (Sub BG).txt',
    r'C:\Users\Juan\Documents\GitHub\spectramap\wintec\outside_CA.txt'
]
glass_file = r'C:\Users\Juan\Documents\GitHub\spectramap\wintec\glass.txt'
n_endmembers = 8

# --- DATA PROCESSING ---

all_matrices = []
file_labels = [os.path.basename(f).split('.')[0] for f in data_files]
sample_info = [] # Store pixel counts and grid sizes

# Load and process glass background
print("Loading glass background...")
glass_data = np.loadtxt(glass_file, delimiter=',', skiprows=1)
glass_wav = glass_data[:, 0]
glass_int = glass_data[:, 1]

processed_wavenumber = None

for i, file_path in enumerate(data_files):
    print(f"\nProcessing {file_labels[i]}...")
    
    # 1. Load Data
    try:
        data = np.loadtxt(file_path, delimiter=',')
    except ValueError:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    wavenumber = data[:, 0]
    matrix = data[:, 1:]
    num_pix = matrix.shape[1]
    
    # Determine grid size (assuming square for images, 1x1 for single spectra)
    current_grid = int(np.sqrt(num_pix)) if num_pix > 1 else 1
    sample_info.append({'label': file_labels[i], 'n_pix': num_pix, 'grid': current_grid})
    
    # 2. Subtract Glass Background
    glass_interp = np.interp(wavenumber, glass_wav, glass_int)
    matrix = matrix - glass_interp[:, np.newaxis]
    
    # 3. Smart Cosmic Ray Removal (only for images)
    if current_grid > 1:
        matrix, n_spikes = remove_cosmic_rays_smart(matrix, current_grid)
        print(f"  - Cosmic rays fixed: {n_spikes}")
    else:
        print("  - Single spectrum detected, skipping 3D cosmic ray removal.")
    
    # 4. Crop Spectral Range (400-3300, skip 1900-2600)
    mask_crop = (wavenumber >= 400) & (wavenumber <= 3300) & ~((wavenumber > 1900) & (wavenumber < 2600))
    wavenumber = wavenumber[mask_crop]
    matrix = matrix[mask_crop, :]
    processed_wavenumber = wavenumber 
    
    # 5. Baseline Correction (airPLS)
    print(f"  - Baseline correction (lambda=1e3) in parallel...")
    # Use all available cores to process pixels in parallel
    baselines = Parallel(n_jobs=-1)(delayed(airPLS)(matrix[:, p], lambda_=1e3) for p in range(matrix.shape[1]))
    matrix = matrix - np.array(baselines).T
    
    # 6. Independent Normalization
    mask_low = (wavenumber <= 1900)
    mask_high = (wavenumber >= 2600)
    
    norms_low = np.linalg.norm(matrix[mask_low, :], axis=0)
    norms_low[norms_low == 0] = 1.0
    matrix[mask_low, :] /= norms_low
    
    norms_high = np.linalg.norm(matrix[mask_high, :], axis=0)
    norms_high[norms_high == 0] = 1.0
    matrix[mask_high, :] /= norms_high
    
    all_matrices.append(matrix)

# Combine all data into one giant matrix
print("\n--- Merging all datasets ---")
master_matrix = np.hstack(all_matrices)
print(f"Master Matrix Shape: {master_matrix.shape}")

# --- VCA UNMIXING ---
print(f"\n--- Running Global VCA with {n_endmembers} endmembers ---")
Ae, indice, _ = vca(master_matrix, n_endmembers, verbose=True)

# --- RECONSTRUCT ABUNDANCES ---
print("\n--- Calculating Abundances (NNLS) in parallel ---")
# Using NNLS to find abundances of global endmembers in every pixel
from scipy.optimize import nnls

def solve_pixel_nnls(pixel, Ae_matrix):
    sol, _ = nnls(Ae_matrix, pixel)
    return sol

num_total_pixels = master_matrix.shape[1]
# Run NNLS deconvolution in parallel across all samples
abundances_list = Parallel(n_jobs=-1)(delayed(solve_pixel_nnls)(master_matrix[:, i], Ae) for i in range(num_total_pixels))
abundances = np.array(abundances_list)
print("Abundance calculation complete.")

# Split abundances back into individual samples
sample_abundances = []
current_pos = 0
for info in sample_info:
    n_pix = info['n_pix']
    grid = info['grid']
    
    if n_pix > 1:
        s_abund = abundances[current_pos:current_pos+n_pix, :].reshape((grid, grid, n_endmembers))
    else:
        s_abund = abundances[current_pos:current_pos+n_pix, :] # Keep 1D for single spectrum
        
    sample_abundances.append(s_abund)
    current_pos += n_pix

#%%
# --- VISUALIZATION ---

# 0. Glass Background Spectrum
plt.figure(figsize=(12, 5))
mask_glass = glass_wav >= 400
plt.plot(glass_wav[mask_glass], glass_int[mask_glass], color='gray', alpha=0.8, label='Glass Background')

# Peak ID for Glass
smoothed_glass = savgol_filter(glass_int[mask_glass], 15, 3)
peaks_glass, _ = find_peaks(smoothed_glass, prominence=np.max(smoothed_glass)*0.05, distance=20, width=3)
wav_subset = glass_wav[mask_glass]
int_subset = glass_int[mask_glass]

for p in peaks_glass[np.argsort(smoothed_glass[peaks_glass])][-8:]:
    plt.text(wav_subset[p], int_subset[p] + (np.max(int_subset)*0.02), f'{wav_subset[p]:.0f}', 
               color='black', fontsize=9, fontweight='bold', ha='center')

plt.title('Glass Background Spectrum (from 400 cm-1)', fontsize=14, fontweight='bold')
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Intensity (a.u.)')
plt.xlim(left=400)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# 1. Global Endmembers Plot
shift_amount = 2600 - 2100
w_display = processed_wavenumber.copy()
w_display[processed_wavenumber >= 2600] -= shift_amount

fig_em, ax_em = plt.subplots(figsize=(12, 6))
offset_step = 1.2
selected_indices = [6, 2, 0] # Corresponds to labels 7, 3, and 1

for idx, i in enumerate(selected_indices):
    ae_viz = Ae[:, i] / np.max(Ae[:, i])
    y_off = ae_viz + idx * offset_step
    line, = ax_em.plot(w_display, y_off)
    
    # Label
    left_idx = np.argmin(w_display)
    ax_em.text(w_display[left_idx], y_off[left_idx] + 0.05, f'Global EM {i+1}', 
               color=line.get_color(), fontweight='bold', ha='left')
    
    # Peak ID
    smoothed = savgol_filter(Ae[:, i], 15, 3)
    peaks, _ = find_peaks(smoothed, prominence=np.max(smoothed)*0.05, distance=20, width=3)
    for p in peaks[np.argsort(smoothed[peaks])][-5:]:
        x_p = processed_wavenumber[p]
        if x_p >= 2600: x_p -= shift_amount
        ax_em.text(x_p, y_off[p] + 0.05, f'{processed_wavenumber[p]:.0f}', 
                   color=line.get_color(), fontsize=9, fontweight='bold', ha='center')

ax_em.text(2000, 0, '//', fontsize=24, fontweight='bold', ha='center')
ax_em.set_title(f'Selected Global Endmembers ({len(data_files)} Samples Combined)', fontsize=14)
ax_em.set_xlabel('Wavenumber (cm-1)')
orig_ticks = np.array([400, 900, 1400, 1900, 2600, 3100])
disp_ticks = orig_ticks.copy()
disp_ticks[orig_ticks >= 2600] -= shift_amount
ax_em.set_xticks(disp_ticks)
ax_em.set_xticklabels([str(t) for t in orig_ticks])
plt.tight_layout()
plt.show()

# 2. Abundance Results per Sample
for s_idx, info in enumerate(sample_info):
    label = info['label']
    
    if info['n_pix'] > 1:
        # Plot 1x3 grid of abundance maps for selected endmembers
        selected_indices = [6, 2, 0]
        fig_maps, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig_maps.suptitle(f'Selected Abundance Maps: {label}', fontsize=16, fontweight='bold')
        
        for idx, i in enumerate(selected_indices):
            im = axes[idx].imshow(sample_abundances[s_idx][:,:,i], cmap='inferno')
            axes[idx].set_title(f'Global EM {i+1}')
            axes[idx].axis('off')
            cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.set_label('Abundance (a.u.)', fontsize=9)
        plt.tight_layout()
        plt.show()
    else:
        # Plot a bar chart for single-point spectra (only selected EMs)
        selected_indices = [6, 2, 0]
        plt.figure(figsize=(8, 4))
        ems = [f'EM {i+1}' for i in selected_indices]
        vals = [sample_abundances[s_idx][0][i] for i in selected_indices]
        plt.bar(ems, vals, color='teal', alpha=0.7)
        plt.title(f'Selected Endmember Contributions: {label}', fontsize=14, fontweight='bold')
        plt.ylabel('Abundance (a.u.)')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

#%%

# --- DATA EXPORT ---
print("\n--- Exporting Global Results ---")
output_dir = os.path.dirname(data_files[0])
selected_indices = [6, 2, 0]

# 0. Export Glass Background Spectrum
glass_export = np.column_stack((wav_subset, int_subset))
glass_header = "Wavenumber,Glass_Intensity"
glass_filename = os.path.join(output_dir, "Glass_Background_Spectrum.csv")
np.savetxt(glass_filename, glass_export, delimiter=",", header=glass_header, comments="")
print(f"Success: Glass background saved to '{os.path.basename(glass_filename)}'")

# 1. Export Selected Global Endmember Spectra
selected_Ae = Ae[:, selected_indices]
em_export = np.column_stack((processed_wavenumber, selected_Ae))
em_header = "Wavenumber," + ",".join([f"Global_EM_{i+1}" for i in selected_indices])
em_filename = os.path.join(output_dir, "Global_VCA_Endmembers_Selected.csv")
np.savetxt(em_filename, em_export, delimiter=",", header=em_header, comments="")
print(f"Success: Selected global endmembers saved to '{os.path.basename(em_filename)}'")

# 2. Export Individual Abundance Maps (Selected Only)
for s_idx, info in enumerate(sample_info):
    label = info['label']
    abund_data = sample_abundances[s_idx]
    
    if info['n_pix'] > 1:
        # Image export: X, Y, EM7, EM3, EM1...
        grid = info['grid']
        x_coords, y_coords = np.meshgrid(np.arange(grid), np.arange(grid))
        # Filter for selected EMs
        selected_abund = abund_data[:, :, selected_indices]
        flat_abund = selected_abund.reshape(-1, len(selected_indices))
        export_matrix = np.column_stack((x_coords.flatten(), y_coords.flatten(), flat_abund))
        
        header = "X_pixel,Y_pixel," + ",".join([f"Global_EM{i+1}_Abundance" for i in selected_indices])
        fname = os.path.join(output_dir, f"Abundance_Map_{label}_Selected.csv")
        np.savetxt(fname, export_matrix, delimiter=",", header=header, comments="")
        print(f"Success: Selected abundance map for '{label}' saved.")
    else:
        # Single spectrum export (Selected Only)
        selected_abund = abund_data[0][selected_indices]
        fname = os.path.join(output_dir, f"Abundance_Point_{label}_Selected.csv")
        header = ",".join([f"Global_EM{i+1}_Abundance" for i in selected_indices])
        np.savetxt(fname, selected_abund, delimiter=",", header=header, comments="")
        print(f"Success: Selected abundance point for '{label}' saved.")

print("\nAll processing and exports complete.")

# %%
