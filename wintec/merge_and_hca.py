#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin
import scipy.sparse
import scipy.sparse.linalg
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.colors import ListedColormap
import os
from joblib import Parallel, delayed

# --- CORE FUNCTIONS ---

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
n_clusters = 10 # Number of HCA clusters (Increased for more detail)
n_pca_components = 15 # PCA for dimensionality reduction before HCA

# --- DATA PROCESSING ---

all_matrices = []
file_labels = [os.path.basename(f).split('.')[0] for f in data_files]
sample_info = []

print("Loading glass background...")
glass_data = np.loadtxt(glass_file, delimiter=',', skiprows=1)
glass_wav = glass_data[:, 0]
glass_int = glass_data[:, 1]

processed_wavenumber = None

for i, file_path in enumerate(data_files):
    print(f"\nProcessing {file_labels[i]}...")
    try:
        data = np.loadtxt(file_path, delimiter=',')
    except ValueError:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    wavenumber = data[:, 0]
    matrix = data[:, 1:]
    num_pix = matrix.shape[1]
    current_grid = int(np.sqrt(num_pix)) if num_pix > 1 else 1
    sample_info.append({'label': file_labels[i], 'n_pix': num_pix, 'grid': current_grid})
    
    glass_interp = np.interp(wavenumber, glass_wav, glass_int)
    matrix = matrix - glass_interp[:, np.newaxis]
    
    if current_grid > 1:
        matrix, n_spikes = remove_cosmic_rays_smart(matrix, current_grid)
        print(f"  - Cosmic rays fixed: {n_spikes}")
    
    mask_crop = (wavenumber >= 400) & (wavenumber <= 3300) & ~((wavenumber > 1900) & (wavenumber < 2600))
    wavenumber = wavenumber[mask_crop]
    matrix = matrix[mask_crop, :]
    processed_wavenumber = wavenumber 
    
    print(f"  - Baseline correction (lambda=1e3) in parallel...")
    baselines = Parallel(n_jobs=-1)(delayed(airPLS)(matrix[:, p], lambda_=1e3) for p in range(matrix.shape[1]))
    matrix = matrix - np.array(baselines).T
    
    # 6. Regional Normalization
    print("  - Applying Independent L2 Normalization (Fingerprint & C-H regions)...")
    mask_low = (wavenumber <= 1900)
    mask_high = (wavenumber >= 2600)
    norms_low = np.linalg.norm(matrix[mask_low, :], axis=0)
    norms_low[norms_low == 0] = 1.0
    matrix[mask_low, :] /= norms_low
    norms_high = np.linalg.norm(matrix[mask_high, :], axis=0)
    norms_high[norms_high == 0] = 1.0
    matrix[mask_high, :] /= norms_high
    
    all_matrices.append(matrix)

print("\n--- Merging all datasets ---")
master_matrix = np.hstack(all_matrices).T # (Pixels, Channels)
print(f"Master Matrix Shape: {master_matrix.shape}")

# --- PCA (Dimensionality Reduction) ---
print(f"\n--- Performing PCA ({n_pca_components} components) ---")
from sklearn.decomposition import PCA
pca = PCA(n_components=n_pca_components)
pca_scores = pca.fit_transform(master_matrix)
print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2%}")

# --- HCA CLUSTERING ---
print(f"\n--- Running Hierarchical Cluster Analysis (Ward's Linkage) ---")
# On large datasets, we use a subset for linkage if necessary, or just run it if memory allows.
# 30,000 pixels is manageable for 'ward' on PCA scores.
Z = linkage(pca_scores, method='ward')
clusters = fcluster(Z, n_clusters, criterion='maxclust')
print(f"Clustering complete. Found {n_clusters} clusters.")

# --- MEAN CLUSTER SPECTRA ---
print("\n--- Calculating Cluster Mean Spectra ---")
cluster_means = []
for i in range(1, n_clusters + 1):
    mask = (clusters == i)
    if np.any(mask):
        mean_spec = np.mean(master_matrix[mask, :], axis=0)
        cluster_means.append(mean_spec)
    else:
        cluster_means.append(np.zeros(master_matrix.shape[1]))

# Split clusters back into individual samples
sample_clusters = []
current_pos = 0
for info in sample_info:
    n_pix = info['n_pix']
    grid = info['grid']
    s_clust = clusters[current_pos:current_pos+n_pix]
    if n_pix > 1:
        s_clust = s_clust.reshape((grid, grid))
    sample_clusters.append(s_clust)
    current_pos += n_pix

# --- VISUALIZATION ---

# 1. Mean Cluster Spectra Plot
shift_amount = 2600 - 2100
w_display = processed_wavenumber.copy()
w_display[processed_wavenumber >= 2600] -= shift_amount

# Create a consistent colormap for both spectra and maps
full_tab20 = plt.get_cmap('tab20')
cluster_cmap = ListedColormap(full_tab20.colors[:n_clusters])

fig_em, ax_em = plt.subplots(figsize=(12, 10))
offset_step = 1.2

for i in range(n_clusters):
    spec = cluster_means[i]
    spec_norm = spec / np.max(spec)
    y_off = spec_norm + i * offset_step
    # Use exact color from our custom map
    line, = ax_em.plot(w_display, y_off, color=cluster_cmap.colors[i])
    
    # Label
    left_idx = np.argmin(w_display)
    ax_em.text(w_display[left_idx], y_off[left_idx] + 0.05, f'Cluster {i+1}', 
               color=cmap(i), fontweight='bold', ha='left')
    
    # Peak ID
    smoothed = savgol_filter(spec, 15, 3)
    peaks, _ = find_peaks(smoothed, prominence=np.max(smoothed)*0.05, distance=20, width=3)
    for p in peaks[np.argsort(smoothed[peaks])][-5:]:
        x_p = processed_wavenumber[p]
        if x_p >= 2600: x_p -= shift_amount
        ax_em.text(x_p, y_off[p] + 0.05, f'{processed_wavenumber[p]:.0f}', 
                   color=cmap(i), fontsize=9, fontweight='bold', ha='center')

ax_em.text(2000, 0, '//', fontsize=24, fontweight='bold', ha='center')
ax_em.set_title(f'Mean Cluster Spectra (HCA)', fontsize=14)
ax_em.set_xlabel('Wavenumber (cm-1)')
orig_ticks = np.array([400, 900, 1400, 1900, 2600, 3100])
disp_ticks = orig_ticks.copy()
disp_ticks[orig_ticks >= 2600] -= shift_amount
ax_em.set_xticks(disp_ticks)
ax_em.set_xticklabels([str(t) for t in orig_ticks])
plt.tight_layout()
plt.show()

# 2. Cluster Maps per Sample
for s_idx, info in enumerate(sample_info):
    label = info['label']
    if info['n_pix'] > 1:
        plt.figure(figsize=(8, 8))
        # Use the same custom colormap for the map
        plt.imshow(sample_clusters[s_idx], cmap=cluster_cmap, interpolation='nearest', 
                   vmin=1, vmax=n_clusters)
        plt.title(f'HCA Cluster Map: {label}', fontsize=14, fontweight='bold')
        cb = plt.colorbar(ticks=np.linspace(1, n_clusters, n_clusters), fraction=0.046, pad=0.04)
        cb.set_label('Cluster ID')
        plt.axis('off')
        plt.show()
    else:
        print(f"Sample {label} is a single point. Assigned to Cluster {sample_clusters[s_idx][0]}")

print("\nHCA Analysis complete. Results displayed.")

# %%
