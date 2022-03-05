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

  <p align="center"> Figure 3 Segmentation by clustering: (a) clustering, (b) image, (c) concentration map and (d) mean clusters

- <p align="justify">Visualization: the next examples shows the pca scores of several biomolecules.

 <p align="center"><img src="https://bl6pap003files.storage.live.com/y4m2IgtZawTrfzKz36eecSGjwkXsjp5Zp5vognNGr-v-VeNX4nLSWbid62R28cW6_gqsxS5JJfNBeF2pzQArOPDEsb3BqTYyyzGo2qA5CuXZaLCER_a6PiwVubWL2B9GB0n6hgHXkSXouTZKLYEHPve_TwUVOtYN9inEhgU3wH5kazukHsbqeyRar4fdgNUg6Bz?width=450&height=501&cropmode=none" /></div>

<p align="center">Figure 4 PCA scores

## Further upcoming developments:
  
  - Graphical User Interface
  - Supervised tools
  - Deep learning - CNN
  - Optimizing speed and organizing main code
  - More examples

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
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mnECDWTSH0PtXtx4Gjc1Vv_Us0gv4T2e9U-bFuSOW6CBbHOGdyvsiCoFeGmYGvDVlsF52sTsKopv63xxTyaXOLQhZk5vd3twL1aAsz9xT-lFr9Qv1WT5aATpPjUOMlg6kV_42FPKbpfoIAdufFmKEWzziLok3n0ngefa2BIynR-UkqHKgpoj0ftX4d3B6EdUd?width=660&height=408&cropmode=none" /></div>

<p align="center">Figure 6 Second visualization

#### Reading and processing a comma separated vector file with depth profiling

<p align="justify">In the <a href="https://github.com/spectramap/spectramap"> examples</a>, there is a layers.csv.xz file for this example. The next lines show some basic tools. The function read_csv requires the path directory of the file. The csv file must keep the structure of the <a href="https://github.com/spectramap/spectramap/blob/main/docs/Manual%20SpectraMap.pdf"> manual </a> (hyperspectral object). The examples shows how to analise the data of spectroscopic profiles.

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

<p align="justify"> In the <a href="https://github.com/spectramap/spectramap"> examples file</a> , there is a layers.csv.xz file for this example. The next processing steps computes unsupervised principal component analysis and double supervised partial least square + linear discriminant analysis. The scatter plots show the separation of the plastics: red, light_blue and blue are the most different ones. 

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
scores_pls, loadings_pls = sample.pls_lda(3, False, 0.7) # 3 components pls-lda  and 70% training data
scores_pls.show_scatter("auto", main_label, sub_label, 15) # showing scatter with sublevel
```
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mXbKMwtXDVcngG68AW4hhLYbPvEsLHZ5Y4hdBuO-JxicuoSZegq-YbNgmNET-kuMC_dW2dqE-CBOQ05FSt29Yx8rT_eeFE_vPyXTxBczgY90b4gChRx3IR3iei0MpERo1yrD6t9hN1TCmGjEzakPU17w8rbMvQ3dbnzV1eBgP-Kol8jlraVtnZHKpTHhWtnf-?width=660&height=431&cropmode=none" /></div>

<p align="justify"> The next figures shows the precision, recall (sensititivity), f1-score (weighted average of preceision) and support for the 6 components. Accuracy and average accuracy. 

<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mj56vCKYH1LPHMS4if0mezHNO5YFzrZxxV626ocQLgfGeXu5eYJNR0fn8_Ap33DKZEURqAUuYvBXrNNm4Qg94-hd598m6cJdBg7w0NWwzNZVFcFuOoJhNUn3aF5T6ARXB_8h8qdodKjYpJhCVQDLvsfN53v5eM__BV_AKYIN2vmK8YuL9TLqqJD8fo6JLyd3V?width=474&height=227&cropmode=none" /></div>

#### Raman wavenumber calibration by paracetaminol

<p align="justify"> Reproducibility and replicativity are fundamental parameters for Raman spectroscopy. One common way for wavenumber axis calibration is discussed in this section. The requirements are a paracetaminol sample (powder) and the calibration file (well-measured peaks) and a polynomial regression.

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
<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mJ89jfrxDF_p5mLQFMbpHL0rsk58_6yoLwOI1_lMk9aT4wMLZyyGSP89l0QfnWipiAZiWDje_UxmuS6uB3LzDHvL7QmnO3ml2dCs4F6pafztjocJLADDlsXVo324KZM2ycI9FyMFLfqMdnumqRwIZpa5VI_uhFsJ8mvHFMUNsStw2OQ3tRZQq0XlQURRozrMN?width=660&height=379&cropmode=none" /></div>

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

#### Raman Intensity Calibration

The next lines show how to calibrate intensity axis in Ramam spectroscopy. It is required a standard spectrum of halogen lamp and the experimental measurement of the halogen lamp with the Raman instrument.

```python
from spectramap import spmap as sp # reading spectramap package
reference_trial = sp.hyper_object("reference") # creating reference hyper object
reference_trial.read_single_spc(path + "reference") # reading the referece data spectrum 
reference_trial.show(True) # showing the spectrum in the next plot
```

<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mS7E9jF96MGDDSPRsp_PO7IJGDX_WVsDNkVU1snc2zNsT9mciLmQ331-BiAAR3_tEDd8a9AANrr-liAzAMbCmAWcAcHDuphckkyu6gCQjChHQy4zm4ISreOfS7CWz-MAwREfhfzkOuUAQG6kRtECnD5tG-rOkOYFHeB49J_93eF0uNo_QKW_Pt5HQmfRnqgI3?width=660&height=401&cropmode=none" /></div>

Now the experimental spectrum.

```python
measured_trial = sp.hyper_object("measured") # creating hyper object
measured_trial.read_single_spc(path + "lamp") # reading data
measured_trial.keep(400, 1900) # keeping finger print region
measured_trial.show(True) # showing the plot as the next figures shows
```

<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mgHiVbaq4PTIwV4-OUf0NhA2HVImJLqbgk_vKGOQ_k7afhEgLaPT0eXU3mCM_VlzELEWSoSE81W_OgM39E_F1gggRt-ILiOSH1jpPbCwtNlSCkjXScUzm0b--LubKANx-zqW5iyklz2f7-axSjnwDi5G-hQpc_21ycVoGtrUp7k_ZGpgETY4iCYSx1iQ5P6TT?width=660&height=401&cropmode=none" /></div>

Reading the Raman sample. 

```python
sample = sp.hyper_object("sample") # declareting hyper object
sample.read_single_spc(path + "sample") # reading tissue data
sample.keep(400, 1900) # keeping finger print region
sample.show(True) # showing plot in the next figure
```

<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mfkYUhxDhwjI0Qel4lqN36GRZpNzyrqBalwxq35lhbz55CJej9k5x5_rtug0DQOEB_lHp7aB5tQfjVlQdw-VfrMUNZgyWSDt-bJq-BxHwB3g2HNtyMBW82iCWFeAW9I4QFAoLNx11gNVQjULOKj9N9EDAONR569qAUCy-qbkTbwyNIQuOk4GSVPD3Tx-IAuVO?width=660&height=401&cropmode=none" /></div>

Calibration of the Raman sample.

```python
sample.intensity_calibration(reference_trial, measured_trial) # intensity calibration function
sample.show(True) # showing the calibrated data in the next figure
```

<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mE8dFFtM-NvQ28s53Md-f9AnhF9rDh0gfn5EBnIin2LRh10eeJJ4cmUZMK4NFTEt7emCIowieDxA2dQ65G4qPtMyeBK0f0kDvtUg7kq7WEibWGL_Z5Wo3FqzSaNWdfnMFNn13dCdzPYSD9Fm17tPsywidPadvWBG4R142LRJ3YEimLDQo_wsPta4-aMh_zm-0?width=660&height=401&cropmode=none" /></div>


## Working Team

<p align="center"><img src="https://bl6pap003files.storage.live.com/y4mHmwP0VTHTFAZZqccQFPVNHS5BTz5fg1mOqqbv_sizMho2majbgupRfZZYl_A1nYzQHXjI5W4T3vgJTKcksjWqe_axT4Ko2-QcEWLgz9YbPn-4qpdbnVFouUPrNza1YS6gV7Kx2_tb_rqxifev3NE-YJIp_vnawgNmEr2eEJcyIQ_Xl-VZNv7qIsh16kl4AKn?width=660&height=161&cropmode=none" /></div>

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
