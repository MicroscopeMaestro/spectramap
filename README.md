<p align="center"><img src="https://bl6pap003files.storage.live.com/y4m8gbPqtqHwQxeiW3C8RLI8BZUy1e3Q3oQ7gJrKNgODSkuJ1_fPVfTtt8J7z6wePajCHMG3lQCk5UXsW1DU4asskXgoAa3h3EH01Zy3eOC5eFgk7gt4Mzk2O-hHYbCh51owTwps0kUFg4umppCPxOCNHgl2AGi_8zwxAwVw0p9Z7yLwRVNN-OopqVXqoMBPYqD?width=660&height=120&cropmode=none" /></div>

## *SpectraMap (SpMap): Hyperspectral package for spectroscopists in Python*

<p align="justify">Hyperspectral imaging presents important current applications in medicine, agriculture, pharmaceutical, space, food and many upcoming applications. The analysis of hyperspectral images requires advanced software. The upcoming developments related to fast hyperspectral imaging, automation and deep learning applications demand innovative software developments for analyzing hyperspectral data. The Figure 1 shows the hyperspectral imaging by a standard spectrometer instrument. More information regarding advances in imaging <a href= "https://advancesimaging.blogspot.com"> advances in imaging!</a> for medical and optics studies.



<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mjeuSGWT6sK3-Q1VrVRR4BrOiazbANDe3408EsJtjx5yE7bPfgREUbfVim-v-0w_45xTiRh1qbbRq5ZMVyPuDIZixrJjFPTt3EPfxtnBcAd9T4ye1TSDBwm8YiH7YRC0gnjreYWl_6-Xwx370cxR1Upso3J6yRzKGOJh9nFnVkAt2_quscBg2nuSuXTSAzaol?width=1920&height=300&cropmode=none" /></div>

<p align="center">Figure 1 Raman Imaging system

## Features

<p align="justify">The package includes standard tools such as reading, preprocessing, processing and visualization. The designing was focused on working hyperspectral images from Raman datasets. The package is extended to other spectroscopies as long as the data follows the type data structure.  Some features are shown by the next figures.

- <p align="justify">Preprocessing: some tool such as smoothing, removal of spikes, normalization and advanced baseline corrections are included. Figure 2 illustrates a mean and standard deviation of a tissue signature.
  
  
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mxWw5ppI-mrsnAVVuXMQmXaSKehSjpUOL9jNFm-2d4UmSVscbu56lkSrBgN0n-I9QKi6leJSqNpvDLGhFqLA9hgXEyqokTieKOxSVFpw_dfjdVaQQAgSSjf9MIQcl7h7iMXfhq5UwA9ZtQDI78AeONLUEs35nZyjgHk6p9ZTs7qJ5VYAKXG4r45SxqWJL3p2b?width=492&height=220&cropmode=none" /></div>

<p align="center"> Figure 2 Visualization of tissue Raman signature

- <p align="justify">Processing: some tools such as unmixing, pca, pls, vca and hierarchical and kmeans clustering are included. Figure 3 displays application of clustering for locating microplastics on complex matrices.

 <p align="center"><img src="https://bl6pap003files.storage.live.com/y4mMByIhOmc82feaIGGCBknJeTWfaUq-xS5hmISMx75_N1UjOz1KdhDEfuvnMF96iI-fMJtfA3nAugSWmp6inEMJTjJSzMBisBk_YYGXBdzP9XMBoZZDylkpRC9kDPyOdSe6v_OZ0iLu3uePduHhjU3I9HEH3LYllxWQ8m5of52yhaMzk_dEomY5tJQ838tw4cp?width=660&height=469&cropmode=none" /></div>
​                                   

  <p align="center"> Figure 3 Segmentation by clustering: (a) clustering, (b) image, (c) concentration map and (d) mean clusters

- <p align="justify">Visualization: the next examples shows the pca scores of several biomolecules.

 <p align="center"><img src="https://bl6pap003files.storage.live.com/y4mJtOVmL6UubzoB0jWAlzO6zoTPBLjayUFBgQXig-TfXagsr5sKEAxDNtHQBLy3L3mvKcdhTrdpppjYOTXpYFoxUTvnEIalaEtrfh0kJwtMDXd6Sbp8MrA_j74VMe5DAehGXbgG1b8lnfBQHOpZqnw3tC7hks8tl5oSKa-IzgHIEbUxwb0y-LUywB-1C6JqmLX?width=660&height=350&cropmode=none" /></div>

<p align="center">Figure 4 PCA scores



## Installation

<p align="justify">The predetermined work interface is Spyder. Install completely anaconda, check the link: https://www.anaconda.com/. The library comes with 5 different hyperspectral examples and analysis. A manual presents the relevant functions and examples (https://spectramap.readthedocs.io/en/latest/).
<p align="justify">Install the library: (admin rights):





```
pip install spectramap
```

## Working Team

Author: Juan-David Muñoz-Bolaños (1)

Contributors: Dr. Tanveer Ahmed Shaik (2), Ecehan Cevik (3), Prof. Jürgen Popp (4) & PD. Dr. Christoph Krafft (5)

(1), (2), (3), (4), (5) Leibniz Institute of photonic technology, Jena, Germany

(1), (3), (4) Friedrich Schiller Jena Universität, Jena, Germany

## License 

<p style="text-align: center;">
    MIT

<p align="justify">Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

<p align="justify">The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

<p align="justify">THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## References

[1] F. Pedregosa, G. Varoquaux, and A. Gramfort, “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825-- 2830, 2011.

[2] J. M. P. Nascimento and J. M. B. Dias, “Vertex component analysis: A fast algorithm to unmix hyperspectral data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 4, pp. 898–910, 2005, doi: 10.1109/TGRS.2005.844293.

[3] Z. M. Zhang, S. Chen, and Y. Z. Liang, “Baseline correction using adaptive iteratively reweighted penalized least squares,” Analyst, vol. 135, no. 5, pp. 1138–1146, 2010, doi: 10.1039/b922045c.

[4] L. McInnes, J. Healy, S. Astels, *hdbscan: Hierarchical density based clustering* In: Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017
