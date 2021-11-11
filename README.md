

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

## Installation

The predetermined work interface is Spyder. Install completely anaconda, check the link: https://www.anaconda.com/. The library comes with 5 different hyperspectral examples and analysis. A manual presents the relevant functions and examples

Moreover, install the following libraries in Anaconda prompt:

```
conda install -c conda-forge hdbscan
pip install pyspectra
pip install spectramap
```

## License

MIT

Author: Juan David Muñoz Bolanos (author)
Contributors: Dr. Tanveer Ahmed Shaik, Ecehan Cevik, PD. Dr. Christoph Krafft

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. © 2021 GitHub, Inc.

