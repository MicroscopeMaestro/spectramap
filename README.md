# SpectraMap (SpMap)

**A Robust Hyperspectral Analysis Package for Spectroscopists in Python**

<p align="justify">
Hyperspectral imaging has critical applications across medicine, agriculture, pharmaceuticals, materials science, and food quality control. <b>SpectraMap</b> provides an advanced, fast, and highly structured framework specifically designed for analyzing Raman and other spectroscopic hyperspectral data.
</p>
<p align="justify">
Recently optimized with a robust architecture, SpectraMap validates data types to ensure reliable and error-free execution of advanced processing techniques like clustering, unmixing, and deep dimensionality reduction.
</p>

---

## 🛠️ Key Features

SpectraMap is built around the highly versatile `hyper_object` class. It includes comprehensive tools covering the entire hyperspectral workflow:

### 1. Robust Data Type Architecture
SpectraMap strictly validates processing operations based on your data type to prevent computational errors:
* `hyper_image`: Full 2D/3D spatial spectral maps (supports spatial mapping, unmixing, and image segmentation).
* `multi_spectra`: Collections of independent spectra without spatial relations (supports PCA, PLS-LDA, scatter plotting).
* `single_spectrum`: Individual spectral signatures (supports baseline correction, smoothing, peak analysis).

### 2. Preprocessing & Correction
Advanced baseline corrections, smoothing algorithms, spike removal, and normalizations.

<p align="center"><img src="https://raw.githubusercontent.com/MicroscopeMaestro/spectramap/main/docs/images/tissue_signature.png" alt="Tissue Signature" /></p>
<p align="center"><i>Figure 1: Mean and standard deviation visualization of a tissue Raman signature.</i></p>

### 3. Machine Learning & Processing
Includes powerful unsupervised and supervised tools:
* **Clustering:** K-Means, Hierarchical (HCA), and Density-Based (DBSCAN/HDBSCAN).
* **Dimensionality Reduction:** Principal Component Analysis (PCA) and Partial Least Squares Discriminant Analysis (PLS-LDA).
* **Unmixing:** Vertex Component Analysis (VCA) for extracting pure endmembers.

<p align="center">
  <img src="https://raw.githubusercontent.com/MicroscopeMaestro/spectramap/main/docs/images/clustering_map.png" alt="Clustering Map" width="45%" />
  <img src="https://raw.githubusercontent.com/MicroscopeMaestro/spectramap/main/docs/images/clustering_stack.png" alt="Clustering Stack" width="45%" />
</p>
<p align="center"><i>Figure 2: Segmentation of microplastics on complex matrices via clustering algorithms.</i></p>

### 4. Advanced Visualization
High-quality rendering of false-color maps, spectral stacks, and scatter profiling.

<p align="center"><img src="https://raw.githubusercontent.com/MicroscopeMaestro/spectramap/main/docs/images/pca_scores.png" alt="PCA Scores" /></p>
<p align="center"><i>Figure 3: 2D PCA scores distinguishing different biomolecular components.</i></p>

---

## 🚀 Installation

SpectraMap is available on PyPI and requires Python 3. 

```bash
pip install spectramap
```

*For comprehensive documentation and all 7 extended hyperspectral examples, please see the [Official Manual](https://github.com/MicroscopeMaestro/spectramap/blob/main/docs/examples.md).*

---

## 📖 Quick Start Guide

### Example 1: Hyperspectral Image (DBSCAN Clustering)
```python
from spectramap import spmap as sp

# Initialize as a hyperspectral image (default)
micro = sp.hyper_object('Microplastics', data_type='hyper_image')
micro.read_csv_xz('examples/microplastics_tissue/microplastics_tissue')

# Apply Hierarchical Density-Based Clustering
micro.dbscan(min_samples=5, epsilon=0.5)

# Render the segmentation map and corresponding spectral stacks
colors = micro.show_map(['gray', 'k', 'r'], None, 1)
micro.show_stack(0, 0, colors)
```

### Example 2: Multi-Spectra (PCA & PLS-LDA)
```python
from spectramap import spmap as sp

# Initialize as independent multiple spectra
plastics = sp.hyper_object("Plastics", data_type="multi_spectra")
plastics.read_csv_xz('examples/plastics_PLS_PCA/plastics')

# Standard preprocessing
plastics.rubber(10)
plastics.vector()

# Principal Component Analysis
scores_pca, loadings_pca = plastics.pca(3)
scores_pca.show_scatter(main_label=15, size=15, colors="auto")
```

---

## 🗺️ Roadmap & Upcoming Developments
- [ ] Interactive Graphical User Interface (GUI)
- [ ] Integration of Supervised Deep Learning models
- [ ] Specialized spectral Large Language Model (LLM) agents
- [x] Robust `data_type` attribute validation framework
- [x] MKDocs comprehensive manual and example gallery

## 📄 License & References
SpectraMap is distributed under the **MIT License**.

**Key Algorithm References:**
1. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 2011.
2. Nascimento & Dias, "Vertex component analysis," *IEEE TGRS*, 2005.
3. Zhang et al., "Baseline correction using adaptive iteratively reweighted penalized least squares," *Analyst*, 2010.
4. McInnes et al., "hdbscan: Hierarchical density based clustering," *JOSS*, 2017.
