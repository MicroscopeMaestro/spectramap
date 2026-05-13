#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize as opt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter

def estimate_snr(Y, r_m, x):
    [L, N] = Y.shape           
    [p, N] = x.shape           
    P_y     = np.sum(Y**2)/float(N)
    P_x     = np.sum(x**2)/float(N) + np.sum(r_m**2)
    snr_est = 10*np.log10( (P_x - p/L*P_y)/(P_y - P_x) )
    return snr_est

def vca(Y, R, verbose=True, snr_input=0):
    if len(Y.shape)!=2:
        raise ValueError('Input data must be of size L by N')
    [L, N] = Y.shape   
    R = int(R)
    if (R<0 or R>L):  
        raise ValueError('ENDMEMBER parameter must be integer between 1 and L')   
    
    if snr_input==0:
        y_m = np.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m           
        Ud  = splin.svd(np.dot(Y_o, Y_o.T)/float(N))[0][:,:R]  
        x_p = np.dot(Ud.T, Y_o)                 
        SNR = estimate_snr(Y, y_m, x_p)
        if verbose:
            print("SNR estimated = {:.2f}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {:.2f}[dB]\n".format(SNR))
    SNR_th = 15 + 10*np.log10(R)
    
    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")
        d = R-1
        if snr_input==0: 
            Ud = Ud[:,:d]
        else:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  
            Ud  = splin.svd(np.dot(Y_o, Y_o.T)/float(N))[0][:,:d]  
            x_p = np.dot(Ud.T, Y_o)                 
        Yp = np.dot(Ud, x_p[:d,:]) + y_m                   
        x = x_p[:d,:] 
        c = np.amax(np.sum(x**2, axis=0))**0.5
        y = np.vstack(( x, c*np.ones((1,N)) ))
    else:
        if verbose:
            print("... Select the projective proj.")
        d = R
        Ud  = splin.svd(np.dot(Y, Y.T)/float(N))[0][:,:d] 
        x_p = np.dot(Ud.T, Y)
        Yp = np.dot(Ud, x_p[:d,:])                 
        x = np.dot(Ud.T, Y)
        u = np.mean(x, axis=1, keepdims=True)        
        y = x / np.dot(u.T, x)
        
    indice = np.zeros((R), dtype=int)
    A = np.zeros((R,R))
    A[-1,0] = 1
    for i in range(R):
        w = np.random.rand(R,1)   
        f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
        f = f / np.linalg.norm(f)      
        v = np.dot(f.T, y)
        indice[i] = np.argmax(np.absolute(v))
        A[:,i] = y[:,indice[i]]        
    Ae = Yp[:,indice]
    return Ae, indice, Yp


file_path = r'C:\Users\Juan\Documents\GitHub\spectramap\wintec\large_CA5.txt'

print("Loading data...")
# Read the CSV data. 
# Try loading normally; if a header exists (like in large_CA3), skip the first row.
try:
    data = np.loadtxt(file_path, delimiter=',')
except ValueError:
    print("Header detected in data file. Skipping first row...")
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)

wavenumber = data[:, 0]
intensity_matrix = data[:, 1:]

# Read and process the glass background spectrum
glass_file = r'C:\Users\Juan\Documents\GitHub\spectramap\wintec\glass.txt'
print("Loading glass spectrum...")
glass_data = np.loadtxt(glass_file, delimiter=',', skiprows=1)
glass_wavenumber = glass_data[:, 0]
glass_intensity = glass_data[:, 1]

# Crop the glass spectrum for the presentation plot (400 - 3300 cm-1)
glass_mask = (glass_wavenumber >= 400) & (glass_wavenumber <= 3300)
glass_wav_crop = glass_wavenumber[glass_mask]
glass_int_crop = glass_intensity[glass_mask]

# Find peaks in the cropped glass spectrum (using smoothing for better detection)
glass_smooth = savgol_filter(glass_int_crop, window_length=15, polyorder=3)
glass_peaks, _ = find_peaks(glass_smooth, prominence=np.max(glass_smooth)*0.005, distance=30, width=5)

# Prepare a shifted wavenumber axis for display to 'join' the two regions with a clear gap
shift_amount = 2600 - 2100  # Increase gap to 200 units to prevent overlapping
glass_wav_display = glass_wav_crop.copy()
glass_wav_display[glass_wav_crop >= 2600] -= shift_amount

# Plot the glass spectrum formatted for PowerPoint
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(glass_wav_display, glass_int_crop, color='#E64A19', linewidth=3.0)

# Annotate the peaks (using display coordinates for X)
if len(glass_peaks) > 0:
    top_glass_peaks = glass_peaks[np.argsort(glass_int_crop[glass_peaks])][-5:]
    for p in top_glass_peaks:
        x_pos = glass_wav_crop[p]
        if x_pos >= 2600: x_pos -= shift_amount
        plt.text(x_pos, glass_int_crop[p] + (np.max(glass_int_crop)*0.03), 
                 f'{glass_wav_crop[p]:.0f}', color='black', fontsize=14, fontweight='bold', ha='center')

# Add the '//' break indicator in the middle of the gap
plt.text(2000, np.min(glass_int_crop), '//', fontsize=24, fontweight='bold', ha='center', va='bottom')

plt.title('Glass Background Spectrum', fontsize=20, fontweight='bold')
plt.xlabel('Wavenumber (cm-1)', fontsize=16, fontweight='bold')
plt.ylabel('Intensity (a.u.)', fontsize=16, fontweight='bold')

# Set custom ticks to show original values without overlapping
orig_ticks = np.array([400, 900, 1400, 1900, 2600, 3100])
disp_ticks = orig_ticks.copy()
disp_ticks[orig_ticks >= 2600] -= shift_amount
plt.xticks(disp_ticks, [str(t) for t in orig_ticks], fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show(block=False)

# Interpolate the glass spectrum to exactly match the image wavenumber axis
glass_interp = np.interp(wavenumber, glass_wavenumber, glass_intensity)

# Subtract the glass spectrum from every pixel in the intensity matrix
print("Subtracting glass background from the dataset...")
intensity_matrix = intensity_matrix - glass_interp[:, np.newaxis]

# %% Smart Cosmic Ray Removal (Spatial-Spectral Consensus)
print("\n--- Removing Cosmic Rays (Smart 3D Method - High Sensitivity) ---")
def remove_cosmic_rays_smart(matrix, grid_size, threshold=4.5):
    """
    Smarter Cosmic Ray Removal (Enhanced):
    - Lower threshold (4.5) for high sensitivity.
    - 5-point spectral median to catch slightly broader spikes.
    - Spatial consensus to prevent clipping real Raman peaks.
    """
    n_channels = matrix.shape[0]
    img_3d = matrix.reshape((n_channels, grid_size, grid_size))
    
    # 1. 5-point Spectral Median Filter
    spec_median = median_filter(img_3d, size=(5, 1, 1))
    spec_diff = img_3d - spec_median
    
    # 2. 3x3 Spatial Median Filter
    spat_median = median_filter(img_3d, size=(1, 3, 3))
    spat_diff = img_3d - spat_median
    
    # 3. Robust Statistics (MAD)
    mad = np.median(np.abs(spec_diff), axis=0)
    mad[mad == 0] = np.mean(mad[mad > 0]) if np.any(mad > 0) else 1.0
    
    # 4. Logic: High sensitivity z-score
    z_score_spec = spec_diff / (1.4826 * mad)
    
    # Refined mask: Spectral outlier AND Spatial outlier
    spike_mask = (z_score_spec > threshold) & (spat_diff > (threshold * 0.5 * mad))
    
    img_clean = img_3d.copy()
    img_clean[spike_mask] = spec_median[spike_mask]
    
    return img_clean.reshape(matrix.shape), np.sum(spike_mask)

intensity_matrix, num_spikes = remove_cosmic_rays_smart(intensity_matrix, grid_size)
print(f"Smart removal complete. Fixed {num_spikes} spike(s) confirmed by spatial-spectral consensus.")

# Calculate spatial dimensions
num_pixels = intensity_matrix.shape[1]
grid_size = int(np.sqrt(num_pixels)) # 10,000 pixels -> 100x100 grid

print("\n--- Cropping Region (400 - 3300 cm-1, ignoring 1900-2600) ---")
# Keep 400-3300, but exclude the "silent" 1900-2600 region to focus on fingerprint and C-H bands
mask_crop = (wavenumber >= 400) & (wavenumber <= 3300) & ~((wavenumber > 1900) & (wavenumber < 2600))
wavenumber = wavenumber[mask_crop]
intensity_matrix = intensity_matrix[mask_crop, :]
print(f"Data cropped to {wavenumber.shape[0]} wavenumbers.")

print("\n--- Savitzky-Golay Smoothing ---")
# Apply a weak filter (window size 5, poly order 3) along the spectral axis (axis 0)
#intensity_matrix = savgol_filter(intensity_matrix, window_length=3, polyorder=3, axis=0)
print("Smoothing complete.")
print("\n--- Baseline Correction (airPLS) ---")

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

def airPLS(x, strength, porder=1, itermax=50):
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, strength, porder)
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

# Apply airPLS baseline correction to all 10,000 pixels
for i in range(num_pixels):
    bg = airPLS(intensity_matrix[:, i], strength=1e3)
    intensity_matrix[:, i] -= bg
    if (i + 1) % 2500 == 0:
        print(f"Processed {i + 1}/{num_pixels} pixels...")
print("Baseline correction complete.")

print("\n--- Independent Region Normalization ---")
# Apply normalization separately to both regions to balance their influence
mask_low = (wavenumber <= 1900)
mask_high = (wavenumber >= 2600)

# Normalize Fingerprint region (<= 1900)
norms_low = np.linalg.norm(intensity_matrix[mask_low, :], axis=0)
norms_low[norms_low == 0] = 1.0
intensity_matrix[mask_low, :] /= norms_low

# Normalize C-H region (>= 2600)
norms_high = np.linalg.norm(intensity_matrix[mask_high, :], axis=0)
norms_high[norms_high == 0] = 1.0
intensity_matrix[mask_high, :] /= norms_high

print("Independent normalization for fingerprint and C-H regions complete.")

# Create a boolean mask for the 600-1800 cm-1 range
mask = (wavenumber >= 600) & (wavenumber <= 1800)

wavenumber_range = wavenumber[mask]
intensity_range = intensity_matrix[mask, :]

# Sort the wavenumbers to ensure the area calculation sign is correct
sort_idx = np.argsort(wavenumber_range)
x_sorted = wavenumber_range[sort_idx]
y_sorted = intensity_range[sort_idx, :]

# Integrate the intensity over the selected wavenumbers for every pixel
try:
    area = np.trapezoid(y_sorted, x=x_sorted, axis=0) # numpy >= 2.0
except AttributeError:
    area = np.trapz(y_sorted, x=x_sorted, axis=0)     # numpy < 2.0

# Reshape the 1D pixel array back into a 2D spatial image
intensity_map = area.reshape((grid_size, grid_size))

#%% Visualization

# Plot the matrix
fig, ax = plt.subplots(figsize=(8, 6))
img_plot = ax.imshow(intensity_map, cmap='viridis')

cbar = fig.colorbar(img_plot, ax=ax)
cbar.set_label('Integrated Intensity')
ax.set_title('Intensity Map (600 - 1800 cm-1)')
ax.set_xlabel('X coordinate (pixels)')
ax.set_ylabel('Y coordinate (pixels)')

plt.show()

print("\n--- Running VCA with 8 endmembers ---")
# VCA expects shape (Channels, Pixels)
# intensity_matrix is (1600, 10000) where 1600 is wavenumbers (Channels) and 10000 is Pixels
Ae, indice, Yp = vca(intensity_matrix, 8, verbose=True)

# Prepare display wavenumber axis to close the 1900-2600 gap with extra space
shift_amount = 2600 - 2100  # Increase gap to 200 units to prevent overlapping
w_display = wavenumber.copy()
w_display[wavenumber >= 2600] -= shift_amount

# Plot the endmembers in a stacked visualization
fig_vca, ax_vca = plt.subplots(figsize=(12, 10))
# Since we normalize each endmember to Max=1 for visualization, we use a fixed offset
offset_step = 1.2 

# Extract the number of endmembers actually calculated
num_ems = Ae.shape[1]

for i in range(num_ems):
    # Normalize each endmember to [0, 1] range for better visual comparison
    ae_viz = Ae[:, i] / np.max(Ae[:, i])
    y_offset_vals = ae_viz + i * offset_step
    
    line, = ax_vca.plot(w_display, y_offset_vals)
    
    # Label the Endmember on the left side
    left_idx = np.argmin(w_display)
    ax_vca.text(w_display[left_idx], y_offset_vals[left_idx] + (offset_step * 0.05), 
                f'Endmember {i+1}', color=line.get_color(), 
                fontweight='bold', fontsize=11, ha='left', va='bottom')
    
    # Automated Peak Identification
    smoothed_Ae = savgol_filter(Ae[:, i], window_length=15, polyorder=3)
    peaks, _ = find_peaks(smoothed_Ae, prominence=np.max(smoothed_Ae) * 0.05, distance=20, width=3)
    
    if len(peaks) > 0:
        top_peaks = peaks[np.argsort(smoothed_Ae[peaks])][-5:]
        for p in top_peaks:
            x_pos = wavenumber[p]
            if x_pos >= 2600: x_pos -= shift_amount
            # Use the normalized plotting values for vertical positioning
            ax_vca.text(x_pos, y_offset_vals[p] + (offset_step * 0.05), 
                        f'{wavenumber[p]:.0f}', 
                        color=line.get_color(),
                        fontsize=9, 
                        fontweight='bold',
                        horizontalalignment='center')

# Add the '//' break indicator in the gap
ax_vca.text(2000, 0, '//', fontsize=24, fontweight='bold', ha='center', va='bottom')

ax_vca.set_title('VCA Endmembers - Stacked Visualization (Normalized for Shape)', fontsize=14)
ax_vca.set_xlabel('Wavenumber (cm-1)', fontsize=12)
ax_vca.set_ylabel('Stacked Intensity (a.u.)', fontsize=12)
ax_vca.grid(axis='x', linestyle='--', alpha=0.3)

# Set custom ticks to show original values without overlapping
orig_ticks = np.array([400, 900, 1400, 1900, 2600, 3100])
disp_ticks = orig_ticks.copy()
disp_ticks[orig_ticks >= 2600] -= shift_amount
ax_vca.set_xticks(disp_ticks)
ax_vca.set_xticklabels([str(t) for t in orig_ticks])

plt.tight_layout()
plt.show()

print("\n--- Calculating Abundances (NNLS) ---")

# M: (N_pixels, L_channels) -> intensity_matrix.T is (10000, 1600)
# U: (R_endmembers, L_channels) -> Ae.T is (8, 1600)
M = intensity_matrix.T
U = Ae.T
N_pixels = M.shape[0]
R_endmembers = U.shape[0]

abundances = np.zeros((N_pixels, R_endmembers), dtype=np.float32)
MtM = np.dot(U, U.T)
for i in range(N_pixels):
    abundances[i] = opt.nnls(MtM, np.dot(U, M[i]))[0]

# Reshape into maps
abundance_maps = abundances.reshape((grid_size, grid_size, R_endmembers))

# Extract the number of endmembers actually calculated
num_ems = abundance_maps.shape[2]
n_cols = (num_ems + 1) // 2  # Calculate columns needed for 2 rows

# Plot the abundance maps
fig_abund, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))
axes = axes.flatten()

for i in range(num_ems):
    im = axes[i].imshow(abundance_maps[:, :, i], cmap='inferno')
    axes[i].set_title(f'Endmember {i+1} Abundance', fontsize=12, fontweight='bold')
    axes[i].axis('off')
    fig_abund.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

# Turn off any empty subplots if the number of endmembers is odd
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# %%
print("\n--- Segmenting Endmember 1 Region ---")
# Extract abundance map for Endmember 1 (index 0)
em1_abundance = abundance_maps[:, :, 0]

# Define a threshold for segmentation (using 85th percentile to get the strongest regions)
threshold = np.percentile(em1_abundance, 85)
em1_mask = em1_abundance > threshold

# Flatten the mask to apply to the linear pixel array
em1_mask_flat = em1_mask.flatten()

# Extract the spectra from the fully preprocessed data matrix (M has shape N_pixels x L_channels)
em1_spectra = M[em1_mask_flat, :]

print(f"Extracted {em1_spectra.shape[0]} spectra from the segmented region.")

# Calculate the mean and standard deviation of these segmented spectra
mean_spectrum = np.mean(em1_spectra, axis=0)
std_spectrum = np.std(em1_spectra, axis=0)

# Plot the segmentation results
fig_seg, axes_seg = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Binary Mask
axes_seg[0].imshow(em1_mask, cmap='gray')
axes_seg[0].set_title('Segmented Region (Endmember 1 > 85th Percentile)')
axes_seg[0].axis('off')

# Plot 2: Extracted Mean Spectrum (using the same split-axis shift)
shift_amount = 2600 - 2100  # Increase gap to 200 units to prevent overlapping
w_display_seg = wavenumber.copy()
w_display_seg[wavenumber >= 2600] -= shift_amount

axes_seg[1].plot(w_display_seg, mean_spectrum, color='crimson')
axes_seg[1].fill_between(w_display_seg, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, 
                         color='crimson', alpha=0.3)
axes_seg[1].set_title('Extracted Spectra from Segmented Region')
axes_seg[1].set_xlabel('Wavenumber (cm-1)')
axes_seg[1].set_ylabel('Intensity')
axes_seg[1].grid(axis='x', linestyle='--', alpha=0.3)

# Add the '//' break indicator in the center of the gap
axes_seg[1].text(2000, np.min(mean_spectrum), '//', fontsize=18, fontweight='bold', ha='center')

# Set custom ticks to show original values without overlapping
orig_ticks = np.array([400, 900, 1400, 1900, 2600, 3100])
disp_ticks = orig_ticks.copy()
disp_ticks[orig_ticks >= 2600] -= shift_amount
axes_seg[1].set_xticks(disp_ticks)
axes_seg[1].set_xticklabels([str(t) for t in orig_ticks])

plt.tight_layout()
plt.show()

# %% Data Export
print("\n--- Exporting Results ---")
import os
# Save results in the same directory as the input file
output_dir = os.path.dirname(file_path)
# Get the name of the folder containing the data to use as a prefix for organization
folder_prefix = os.path.basename(output_dir)

# 1. Export Endmember Spectra (Wavenumber + 10 Components)
# Combine wavenumber and endmembers into one matrix
endmembers_export = np.column_stack((wavenumber, Ae))
# Create header
em_header = "Wavenumber," + ",".join([f"Endmember_{i+1}" for i in range(Ae.shape[1])])
em_filename = os.path.join(output_dir, f"{folder_prefix}_vca_endmember_spectra.csv")
np.savetxt(em_filename, endmembers_export, delimiter=",", header=em_header, comments="")
print(f"Success: Endmember spectra saved to '{os.path.basename(em_filename)}'")

# 2. Export Abundance Maps (Long Format: X, Y, EM1... EM10)
# Create spatial coordinate grids
x_coords, y_coords = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# Flatten the 3D abundance maps back to 2D (Pixels x Components)
abund_flat = abundance_maps.reshape(-1, abundance_maps.shape[2])
# Combine X, Y coordinates with the abundance data
coords_flat = np.column_stack((x_coords.flatten(), y_coords.flatten()))
abund_export = np.column_stack((coords_flat, abund_flat))

# Create header for abundances
abund_header = "X_pixel,Y_pixel," + ",".join([f"EM{i+1}_Abundance" for i in range(abundance_maps.shape[2])])
abund_filename = os.path.join(output_dir, f"{folder_prefix}_vca_abundance_maps.csv")
np.savetxt(abund_filename, abund_export, delimiter=",", header=abund_header, comments="")
print(f"Success: Abundance maps saved to '{os.path.basename(abund_filename)}'")

print("\nAll processing complete. Results are ready for presentation.")

# %%
