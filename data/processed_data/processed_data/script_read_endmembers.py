#%%
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Load Data & Parse Header Metadata
# -----------------------------------------------------------------------------
file_path = 'G:\Raman\processed_data\processed\endmember_spectra.csv'

# Read metadata lines starting with '#'
metadata = []
with open(file_path, 'r') as f:
    for line in f:
        if line.startswith('#'):
            metadata.append(line.strip())
        else:
            break

print("--- Header Metadata ---")
for meta in metadata:
    print(meta)

# Read numerical data into DataFrame (ignoring lines starting with '#')
df = pd.read_csv(file_path, comment='#')

wavenumber = df['Wavenumber']
endmembers = [col for col in df.columns if col != 'Wavenumber']

# -----------------------------------------------------------------------------
# 2. Option A: Overlaid Spectra Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5), dpi=150)

for col in endmembers:
    plt.plot(wavenumber, df[col], label=col, linewidth=1.2)

plt.title('Raman Endmember Spectra (Overlaid)', fontsize=12, fontweight='bold')
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
plt.ylabel('Normalized Intensity (a.u.)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()

# Save and display overlaid plot
plt.savefig('endmember_spectra_overlaid.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 3. Option B: Grid of Individual Subplots
# -----------------------------------------------------------------------------
num_endmembers = len(endmembers)
fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, dpi=150)
axes = axes.flatten()

for i, col in enumerate(endmembers):
    axes[i].plot(wavenumber, df[col], color='navy', linewidth=1.2)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    
    # Label x-axis for bottom plots
    if i >= num_endmembers - 2:
        axes[i].set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=10)
        
    # Label y-axis for left-column plots
    if i % 2 == 0:
        axes[i].set_ylabel('Intensity (a.u.)', fontsize=10)

plt.suptitle('Raman Endmember Spectra (Individual Subplots)', fontsize=13, fontweight='bold')
plt.tight_layout()

# Save and display grid plot
plt.savefig('endmember_spectra_subplots.png', dpi=150)
plt.show()