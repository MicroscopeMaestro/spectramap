# Examples

This section provides various examples demonstrating how to use the `spectramap` library to load, process, and analyze hyperspectral data.

## 1. 3D Volume
Demonstrates loading a 3D hyperspectral cube, pre-processing, and estimating concentrations using Vertex Component Analysis (VCA) and Non-Negative Least Squares (NNLS).

```python
from spectramap import spmap as sp
import pandas as pd

# Creating the hyper object
cube = sp.hyper_object('cube') 

# Reading the 3d csv file and placing resolutions xy = 60 um and z = 70 um
cube.read_csv_3d_xz('examples/3D/cube') 

# Preprocessing
cube.keep(500, 1800) # fingerprint selection
cube.airpls(100) # advanced baseline correction
cube.vector() # vector normalization

# Processing
vca = cube.vca(4) # number of expected components
vca.show_stack(0, 0, 'auto')

# Concentration estimation by NNLS
abundance = cube.abundance(vca, 'NNLS') 
aux = abundance.show_intensity_volume(0.5) # 3d plot of all clusters
```

## 2. Bladder Tissue
Demonstrates loading hyperspectral data of bladder tissue, pre-processing, and performing K-means clustering.

```python
from spectramap import spmap as sp

bladder = sp.hyper_object("bladder")
bladder.read_csv_xz("examples/bladder/bladder")

bladder.set_resolution(0.3) ## 300 um step size resolution
bladder.vector() # vector normalization
original = bladder.get_data() ## get data

# K-means clustering: 3 components
bladder.kmeans(3) 
bladder.remove_label([1])
colors = bladder.show_map(['black', 'green'], None, 1) 
bladder.show_stack(0, 0, colors) # show stack of the clusters
```

## 3. Layers
Demonstrates loading hyperspectral data of layers, pre-processing, estimating endmembers via VCA, estimating abundance, and displaying a profile.

```python
from spectramap import spmap as sp

# Reading
stack = sp.hyper_object('layers')
stack.read_csv_xz('examples/layers/layers')

# Preprocessing
stack.keep(500, 1800) # fingerprint selection
stack.rubber() # baseline correction
stack.vector() # vector normalization

# Processing
endmember = stack.vca(6) # vertex component analysis
endmember.show_stack(0.3, 0, 'auto') # visualization of spectra and strong peaks
abundance = stack.abundance(endmember, 'NNLS') # concentration estimation by NNLS
abundance.set_resolution(0.02) # set resolution of 20 um for the profile
abundance.show_profile('auto') # plot profile
```

## 4. Microplastics Tissue
Demonstrates loading hyperspectral data of microplastics in tissue and applying hierarchical density-based clustering (HDBSCAN).

```python
from spectramap import spmap as sp

# Reading
micro = sp.hyper_object('MP')
micro.read_csv_xz('examples/microplastics_tissue/microplastics_tissue')

# Processing
micro.hdbscan(5, 5) # hierarchical density-based clustering
colors = micro.show_map(['gray', 'k', 'r'], None, 1) # 2D map of the clusters
micro.show_stack(0, 0, colors) # stack of the spectral clusters
```

## 5. Plastics Concatenation
Demonstrates loading multiple single-spectrum samples, computing their means, concatenating them, and visualizing.

```python
from spectramap import spmap as sp

red = sp.hyper_object('red')
red.read_csv_xz('examples/plastics/red')
red.set_label('red')
meanred = red.mean()

blue = sp.hyper_object('blue')
blue.read_csv_xz('examples/plastics/blue')
blue.set_label('blue')
meanblue = blue.mean()

natural = sp.hyper_object('natural')
natural.read_csv_xz('examples/plastics/natural')
natural.set_label('natural')
meannatural = natural.mean()

# Concatenating the three hyperspectral objects
concat = sp.hyper_object('concat')
concat.concat([meanred, meanblue, meannatural])
concat.show_stack(0, 0, ['red', 'blue', 'gray'])
```

## 6. PCA and PLS-LDA on Plastics
Demonstrates applying Principal Component Analysis (PCA) and Partial Least Squares Discriminant Analysis (PLS-LDA).

```python
from spectramap import spmap as sp

plastics = sp.hyper_object('plastics')
plastics.read_csv_xz('examples/layers/layers')

plastics.keep(400, 1850) # keeping fingerprint and high wavenumber region
plastics.gaussian(2) # applying gaussian filter
plastics.rubber() # rubber baseline correction
plastics.vector()

# K-means clustering for labels
plastics.kmeans(6) 
main_label = plastics.get_label()
main_label.name = "main_label"
plastics.show_stack(0, 0, "auto")

# PCA
scores_pca, loadings_pca = plastics.pca(3, False)
scores_pca.show_scatter(main_label, 15, "auto")

# PLS-LDA
scores_pls, loadings_pls = plastics.plslda(3, 1)
scores_pls.show_scatter(main_label, 15, "auto")
```
