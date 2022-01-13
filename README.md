<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mndHVHqnpqVudLMiDZbho2Gi45g9sNnh1OzVkAY0MOwWH9Fm5keWXSrxCdfAum-K4yijPD3dSUTbxPHVeI9OHoa-EkMWfGn2d4XARNHqiGBVr25fCJUx0IWYZYgrDnW2nGtS0PuPDR1M-KvSmoSnC5tNuqH_KatsV68MFPr984_eUQWGk0GEjd5vtvpafqrGN?width=500&height=394&cropmode=none"</div>

## *SpectraMap (SpMap): Hyperspectral package for spectroscopists in Python*

<p align="justify">Hyperspectral imaging presents important current applications in medicine, agriculture, pharmaceutical, space, food and many upcoming applications. The analysis of hyperspectral images requires advanced software. The upcoming developments related to fast hyperspectral imaging, automation and deep learning applications demand innovative software developments for analyzing hyperspectral data. The Figure 1 shows the hyperspectral imaging by a standard spectrometer instrument. More information regarding novel medical imaging is found in <a href= "https://advancesimaging.blogspot.com"> advances in imaging</a>.

<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mFDcOdm5472CEwuu-1aCTB20ZzS5wxLSO9bMZer1YgIQE2ekouGnfET2yuRF4jQbr9MoxPhw4FLX7ZbpBTF4vrYUnnMK3WP3_bQg7oyFdxTTYJmX7bSvu6k3gjZoWJL2wToqf52Ga3dopLGuaGXqxu4LHhQjot9_8yGPowpjisnI8vpPQ-7URYfgRNNH5oJ8S?width=660&height=371&cropmode=none" /></div>

<p align="center">Figure 1 Raman Imaging system

## Features

<p align="justify">The package includes standard tools such as reading, preprocessing, processing and visualization. The designing was focused on working hyperspectral images from Raman datasets. The package is extended to other spectroscopies as long as the data follows the type data structure.  Some features are shown by the next figures.

- <p align="justify">Preprocessing: some tool such as smoothing, removal of spikes, normalization and advanced baseline corrections are included. Figure 2 illustrates a mean and standard deviation of a tissue signature.
  
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mplHFW8SLZNdnUpXqO6g4scKHgzE0F2HF-24bwCf5qTiZnX-S1WjV95CU_8PFufzzf2PeQewZTcuUyhAuFpOyMub5NCail6phkrkXjpldosPcdwTFOpAFhq8i0stGiEoUETcUKvnSMBFVp_R7bKl66-vU36itVQl5hdAntSP71hJ6qMXPbtDmnWacYo-YdBro?width=550&height=400&cropmode=none" /></div>

<p align="center"> Figure 2 Visualization of tissue Raman signature

- <p align="justify">Processing: some tools such as unmixing, pca, pls, vca and hierarchical and kmeans clustering are included. Figure 3 displays application of clustering for locating microplastics on complex matrices.

 <p align="center"><img src="https://bl6pap003files.storage.live.com/y4mCv3oo8wnXEf1lEJiK01NOUET8Sbt3yMIlkReJ3CsKhBV2yaVJ43ZLUFEhW0i7vGdLAagLDJAlomRYrutpLl2mbg8oxa5QPCmHjP2Ktz1dzoRtkroi8vJWCtA67hbCC6sElL0LvyyKhwao7ZhqE5TZQQA_EV-tl3qctMSOalqcREcFyTXiULJXz-FtlpEBZdD?width=660&height=574&cropmode=none" /></div>
​                                   
  <p align="center"> Figure 3 Segmentation by clustering: (a) clustering, (b) image, (c) concentration map and (d) mean clusters

- <p align="justify">Visualization: the next examples shows the pca scores of several biomolecules.

 <p align="center"><img src="https://bl6pap003files.storage.live.com/y4m2IgtZawTrfzKz36eecSGjwkXsjp5Zp5vognNGr-v-VeNX4nLSWbid62R28cW6_gqsxS5JJfNBeF2pzQArOPDEsb3BqTYyyzGo2qA5CuXZaLCER_a6PiwVubWL2B9GB0n6hgHXkSXouTZKLYEHPve_TwUVOtYN9inEhgU3wH5kazukHsbqeyRar4fdgNUg6Bz?width=660&height=501&cropmode=none" /></div>

<p align="center">Figure 4 PCA scores


## Installation

<p align="justify">The predetermined work interface is Spyder. Install completely anaconda, check the link: <a href="https://www.anaconda.com"> Anaconda </a>. The library comes with 4 different hyperspectral examples and analysis. A manual presents the relevant functions and examples <a href="https://spectramap.readthedocs.io/en/latest"> Manual</a>.
<p align="justify">Install the library: (admin rights):


```python
pip install spectramap
```

## Examples

#### Reading and processing a spc file

In the <a href="https://github.com/spectramap/spectramap"> examples file</a> , there is ps.spc file for this example. The next lines show some basic tools. The function read_single_spc reads the path directory of the file.

```python
from spectramap import spmap as sp #reading spmap
pigm = sp.hyper_object('pigment') #creating the hyperobject
pigm.read_single_spc('pigment') #reading the spc file
pigm.keep(400, 1800) #Keeping fingerprint region
pigm_original = pigm.copy() #Copying hyperobject
pigm_original.set_label('original') #renaming hyperobject to original
pigm.set_label('processed') #renaming hyperobject to processed

pigm.rubber() #basic baseline correction rubber band
pigm.gol(15, 3, 0) #savitzky-golay filter
both = sp.hyper_object('result') #creating an auxilary hyperobject
both.concat([pigm_original, pigm]) #concatenating the original and processed data
both.show(False) #show both spectra 
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4m-pe9JbCoAZrJW-nBGBe4LGPLALTafIo3ZJPznScF9felxCXxVSLdA83DGLCKy_wlIj37r8UXBFWlgh1P0imLcFbEvveTJ46j4japWXklN8qttiM3X_y1Hid1YmANAq9EJS0crhltOFXjQt39S0ofUbHqQ0NxgF449sw8NUG92xTLjBq3B1niaUk7S4-qYg47?width=660&height=408&cropmode=none"  /></div>

```python
both.show_stack(0.2, 0, 'auto') #advanced stack visualization 
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mNB6Emvcdx8u4SMExM_oq_O9YPnuAjNCspMxWVzE1rgmIQvjLx-HYL6fmRPdq6WOnNMM8CvtucqbqKoxoMy11oW4dn2TL8--eiqHF_AXAF1eZ31TOz56hjAfIJIL71RHCGObtI-mAR4lC32Vqb_kC56Xqy_qigzq0lBftU6WmLmsiQcV6EisLAKpU4VX_3wOM?width=600&height=769&cropmode=none" /></div>

<p align="center">Figure 6 Second visualization

#### Reading and processing a comma separated vector file with depth profiling

In the <a href="https://github.com/spectramap/spectramap"> examples file</a> , there is a layers.csv.xz file for this example. The next lines show some basic tools. The function read_csv requires the path directory of the file. The csv file must keep the structure of the manual (hyperspectral object). The examples shows how to analise the data of spectroscopic profiles.

```python
from spectramap import spmap as sp # reading spectramap library
stack = sp.hyper_object('plastics') # creating the hyper_object
stack.read_csv_xz('layers') # reading compressed csv of plastics profile
stack.keep(500, 1800) # keeping fingerprint region
stack.rubber() # baseline correciton rubber band
stack.vector() # vector normalization
endmember = stack.vca(6) # number of endmembers  
endmember.show_stack(0.2, 0, 'auto') # advanced stack plot of endmembers 
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4m30AdeakA3242L2iGqNBf75gkIgpNdA1SWwgV3I2bq41q8oOZ0wiVkrRSw9-z-D3sbsLA6aBBZZuyQ01JkzdebzEoEuxcWmbzRj7EvnTRjSJDYYjyY1y5oiU3-G4iolIqAtjiEmqVtAzmzPMw2KOqIUxPQB-n9JoK4xbX24_Krql4TiwhU-2rTSyg_VF6wI8M?width=660&height=408&cropmode=none" /></div>

```python
abundance = stack.abundance(endmember, 'NNLS') # estimation of concentrations by NNLS
abundance.set_resolution(0.01) # setting the step size resolution
abundance.show_profile('auto') # plotting spectral profile 
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mLDJEXyCgxNqLUYQkDAD8qKZWBF8PGpvEz0oX-Iie6TdKMACBpc1Bl4EqZwSfIoLVNWnstFK_q36k5RY-lJHQtAyr8or_TOMetowWHjrdc6xipY8PSbeSSDrXeE7YoKTa0xVCqZraJ5ec-ySyYd01cdFi4k_XTq-etSZGq8uJQf5WQHoiV0IYjEmYWJ0izLd_?width=660&height=408&cropmode=none" /></div>

#### Processing hyperspectral images by VCA and Clustering

comming soon. For now on, Check the manual.

#### Processing plastics hyperspectral data by PCA and PLS-LDA

In the <a href="https://github.com/spectramap/spectramap"> examples file</a> , there is a layers.csv.xz file for this example. The next processing steps computes unsupervised principal component analysis and double supervised partial least square + linear discriminant analysis. The scatter plots show the separation of the plastics: red, light_blue and blue are the most different ones. 

```python
from spectramap import spmap as sp # reading spectramap library
sample = sp.hyper_object("sample") # creating hyper_object
sample.read_csv_xz("layers") # reading compressed csv of plastics profile
sample.remove(1800, 2700) # removing silent region
sample.keep(400, 3300) # keeping finger print and high wavenumber region
sample.gaussian(2) # appliying gaussian filter
sample.rubber() # rubber baseline correction
sample.kmeans(2) # kmeans 2 clusters
sample.rename_label([1, 2], ["first", "second"]) # rename labels
sub_label = sample.get_label()  # saving sub_labels
sub_label.name = "sub_label" # renaming the title of sub_label
sample.show_stack(0,0, "auto") # showing a stack
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4m1LwvuTidoiBCa9enm-_OaENR3KfWSFkgyHGANrk2ii-uY9-vWwmWF5fjSM9dF-H9w-O0TOTfR3MWh8lmVOIN5iHwhb7UxcI6nzHHdAwLucGaXEKMuXVktgZ83eYljUHmCwzRhAfevqW63EWywF0WgBnvw_XRribVVREalh9XS9Eoe5IE9thY9hd_f3utuIvx?width=660&height=410&cropmode=none" /></div>

```python
sample.kmeans(6) # kmeans clustering example for main_label
main_label = sample.get_label() # saving the main_label
main_label.name = "main_label" # renaming the title of the label
sample.show_stack(0,0, "auto") # showing the 6 components
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mYlaMwL3OhKuch2tv4XR_4q9o0EoGK6sn-I6PwKjtvkqJECcDAmz76rLSXdMw_v86tLSKltHM756ULIpkkpuOZO8s3ATOUkzsgzWakF7JShfxlBOUFp-vgexi33aID4Jj6NzxBVGZSUdFFPaAhTxJLg7oUJwkgapfoBpzg1mT89uTUC4dHqEXG5XTbLyLonD5?width=660&height=410&cropmode=none" /></div>

```python
scores_pca, loadings_pca = sample.pca(3, False) # 3 components pca
scores_pca.show_scatter("auto", main_label, sub_label, 15) # showing scatter with sublabel
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mXLVSKKaTqp_chimU_qvxTGXiNIq0LegkAQmVHTrkjQ4nIXICALBisIk2bQNmMaGgVQGEkAdmoYQaKuH-bXVgDMDRGct9_9cW5ABHOVsx-aYmbXQKtcHYLZNwT8Kz7PFqmQkuZBkzM5dmjfjkK0N4AxTSl4OM2XRHbwaUqflvLzH-UWF7Ts4IpowpphDU2Zwx?width=660&height=426&cropmode=none" /></div>

```python
scores_pls, loadings_pls = sample.pls_lda(3, False, 1) # 3 components pls-lda 
scores_pls.show_scatter("auto", main_label, sub_label, 15) # showing scatter with sublevel
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mXbKMwtXDVcngG68AW4hhLYbPvEsLHZ5Y4hdBuO-JxicuoSZegq-YbNgmNET-kuMC_dW2dqE-CBOQ05FSt29Yx8rT_eeFE_vPyXTxBczgY90b4gChRx3IR3iei0MpERo1yrD6t9hN1TCmGjEzakPU17w8rbMvQ3dbnzV1eBgP-Kol8jlraVtnZHKpTHhWtnf-?width=660&height=431&cropmode=none" /></div>

#### Raman wavenumber calibration by paracetaminol

Reproducibility and replicativity are fundamental parameters for Raman spectroscopy. One common way for wavenumber axis calibration is discussed in this section. The requirements are a paracetaminol sample (powder) and the calibration file (well-measured peaks) and a polynomial regression.
```python
from spectramap import spmap as sp # reading spectramap library
import pandas as pd
import numpy as np

### Paracetaminol 
path = 'para.csv' # path of the paracetaminol data
table = pd.read_table(path, sep = ',', header = None) # read data
table['label'] = "Para" # create label
table[['x', 'y']] = np.zeros((20,2)) # create fake positions
### Processing
mp = sp.hyper_object("Para") # creation of hyper object
mp.set_data(table.iloc[:,:len(table.columns)-3]) # reading the intensity 
mp.set_position(table[['x', 'y']]) # reading positions
mp.set_label(pd.Series(table['label'])) # reading labeling
copy = mp.copy() # copy data
peaks = copy.calibration_peaks(mp, 0.05) # finding peaks of para (next plot)
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4m7uHZQOKtk7BEfk6rQ2dFW1MjLzB66kObH4c-Bdlo-l6SfPHQ9tvDJHi2I-m-daTS7hq9IbCWH-bfcwV8sBh17izoUfPszYzUN8Xbt1ULIIZ9ZGGEfMB9H2vAWuo8X3mSeWmVpqzwId-E3cuj3e6BPLhNSFwwAHMhTxrCd9XzDdWaKs5loejkkFhqA96Q6f-Y?width=660&height=379&cropmode=none" /></div>

```python
copy.calibration_regression(peaks) # determining regression for the calibration
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mKTK3bLA_XkVa6pTbs_x7VFxmOsiLI9hmczm-77fwTnKfOYaF4UiP_BZAVnQZkJP2kDilM-dlCLBYnvncwh6eBfGJvdvt9rYxvghpaztSNHX7kCAGphEUJEQarK_OdaGNU11tAUEACOn1mavJrK-v8W-Jdbdg_367GmcO2CpxLpiDgT8PGtDNHQkPsGrs5kkT?width=660&height=379&cropmode=none" /></div>

```python
mp.set_wavenumber(copy.get_wavenumber()) # set the new wavenumber to the original mp
mp.show(True) # show calibrated data
mp.add_peaks(0.1, 'r') # add peaks (not inline mode)
mp.save_data("", "calibration") # save calibrated data
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4m8GE0Ho0DvmWdXyE0ZN_aqpVjWxdma-c_Ll_QD9R20Ke9B9dEinZR8er0Fjg3HMJRVcv_AYsqHNyWUKevvBeG6nj07EE9vIWDC2zr3uaIpvMcJKhnF_Rcu1zpxmymWqiCnNLlIpeTnxGnXwW734ZVQjb9mCEfniU-aXDgufoLvFEgQgWgTs4w7e3XOG6nWrur?width=660&height=324&cropmode=none" /></div>

#### Processing hyperspectral images from biological tissue

comming soon. For now on, Check the manual.

## Working Team

Author: Juan-David Muñoz-Bolaños (1)

Contributors: Dr. Tanveer Ahmed Shaik (2), Ecehan Cevik (3), Prof. Jürgen Popp (4) & PD. Dr. Christoph Krafft (5), (6) Shivani 

(1), (2), (3), (4), (5), (6) Leibniz Institute of photonic technology, Jena, Germany

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
