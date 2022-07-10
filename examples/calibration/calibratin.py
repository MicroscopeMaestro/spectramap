# -*- coding: utf-8 -*-
"""
Created on Mon May 30 19:51:53 2022

@author: juand
"""
from spectramap import spmap as sp # reading spectramap library
### Paracetaminol
mp = sp.hyper_object("para")
#read data with columns in pixels
mp.read_csv_xz("para")

copy = mp.copy() # copy data
peaks = copy.calibration_peaks(mp, 0.05) # finding peaks of para (next plot)
copy.wavenumber_calibration(peaks) # determining regression for the calibration
mp.set_wavenumber(copy.get_wavenumber()) # set the new wavenumber to the original
mp.keep(300, 1850)
mp.show2(True, 0.1, "r") # add peaks (not inline mode)

reference_trial = sp.hyper_object("reference") # creating reference hyper object
reference_trial.read_single_spc("reference") # reading the lamp referece data along wavelength
reference_trial.show(True) # showing the spectrum in the next plot
measured_trial = sp.hyper_object("measured") # creating hyper object
measured_trial.read_single_spc("lamp") # reading lamp experimental data
measured_trial.keep(400, 1900) # keeping finger print region
measured_trial.show(True) # showing the plot as the next figures shows

sample = sp.hyper_object("sample")
sample.read_single_spc("sample")
sample.keep(400, 1900)
sample.intensity_calibration(reference_trial, measured_trial) # intensity calibration function
sample.show(True) # showing the calibrated data in the next figure