

# Spectramap (SpMap)   ![logo](https://bl6pap003files.storage.live.com/y4ma9CnXYI2wnm6MzgvXzMsb5d3Xl36HrNrgl7Gr9MH4PhgECGXRcQln7pps-B7OwyuzgLR2BqcVB3ToIu3GolLP47hRxb1rAGGF6bMpvGyfN0laC04c775Gciuf8KYTd-NfsQwF4CpWAk0fU8SOpwnTKTwkEGb5vZdplqVFwwoo05QqiDSDxBY7n8heezkiaLG?width=660&height=140&cropmode=none)

## *Hyperspectral package for spectroscopists in Python*

Hyperspectral imaging presents important current applications in medicine, agriculture, pharmaceutical, space, food and many upcoming applications. The analysis of hyperspectral images requires advanced software. The upcoming developments related to fast hyperspectral imaging, automation and deep learning applications demand innovative software developments for analyzing hyperspectral data. Along this chapter a new software is development based on Python 3. The library is defined as spectramap.

## Features

* Preprocessing: smoothing, removal of spikes, normalization and advanced baseline corrections

  ![](https://bl6pap003files.storage.live.com/y4mxWw5ppI-mrsnAVVuXMQmXaSKehSjpUOL9jNFm-2d4UmSVscbu56lkSrBgN0n-I9QKi6leJSqNpvDLGhFqLA9hgXEyqokTieKOxSVFpw_dfjdVaQQAgSSjf9MIQcl7h7iMXfhq5UwA9ZtQDI78AeONLUEs35nZyjgHk6p9ZTs7qJ5VYAKXG4r45SxqWJL3p2b?width=492&height=220&cropmode=none)

  ​	Visualization of tissue Raman signature

* Processing: unmixing, pca, pls, vca and hierarchical and kmeans clustering

  ![clustering](https://bl6pap003files.storage.live.com/y4mMByIhOmc82feaIGGCBknJeTWfaUq-xS5hmISMx75_N1UjOz1KdhDEfuvnMF96iI-fMJtfA3nAugSWmp6inEMJTjJSzMBisBk_YYGXBdzP9XMBoZZDylkpRC9kDPyOdSe6v_OZ0iLu3uePduHhjU3I9HEH3LYllxWQ8m5of52yhaMzk_dEomY5tJQ838tw4cp?width=5529&height=3926&cropmode=none)

  ​	Segmentation by clustering

* Visualization

  ![stack](https://bl6pap003files.storage.live.com/y4mRRZQx0bt-IIBFXObjtlov0KAy_K-fSTvOJg8iRyPSjUfAX9apbQX3o7CCfxFMjyGLWzmPtHLLYu4TLMYRh89mLihwohILYusFf5ub6BhmKRaqMbV_asmaHM9PPGA9GgM51E3ZQEWulXdm0pea4pFsKDaY4mKp1ZQJ4JOxUwexWKSAYCR51IOjYFvRanqHyi_?width=660&height=450&cropmode=none)

## References

[1] F. Pedregosa, G. Varoquaux, and A. Gramfort, “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825-- 2830, 2011.

[2] J. M. P. Nascimento and J. M. B. Dias, “Vertex component analysis: A fast algorithm to unmix hyperspectral data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 4, pp. 898–910, 2005, doi: 10.1109/TGRS.2005.844293.

[3] Z. M. Zhang, S. Chen, and Y. Z. Liang, “Baseline correction using adaptive iteratively reweighted penalized least squares,” Analyst, vol. 135, no. 5, pp. 1138–1146, 2010, doi: 10.1039/b922045c.

[4] L. McInnes, J. Healy, S. Astels, hdbscan: Hierarchical density based clustering In: Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017

## Installation

The predetermined work interface is Spyder. Install completely anaconda, check the link: https://www.anaconda.com/. The library comes with 5 different hyperspectral examples and analysis. A manual presents the relevant functions and examples

Moreover, install the following libraries in Anaconda prompt:

```
conda install -c conda-forge hdbscan
pip install pyspectra
pip install spectramap
```

## License

GNU V3

Author: Juan David Muñoz Bolanos

Contributors: Dr. Tanveer Ahmed Shaik, Ecehan Cevik, PD. Dr. Christoph Krafft

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

