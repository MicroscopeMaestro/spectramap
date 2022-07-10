# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:31:23 2022

@author: juand
"""

from spectramap import spmap as sp
### reading ###
path = 'examples/layers/layers'
stack = sp.hyper_object('layers') #creating the hyper object
stack.read_csv_xz(path) #reading the csv file Wavenumber (cm-1)
### preprocessing ###
stack.keep(500, 1800) #finger print selection
stack.rubber() #baseline correction
stack.vector() #vector normalization
### processing ###
endmember = stack.vca(6) # vertex component analysis
endmember.show_stack(0.3, 0, 'auto') # visualization of spectra and strong peaks(Fig.5.6(a))
abundance = stack.abundance(endmember, 'NNLS') # concentration estimation by NNLS
abundance.set_resolution(0.02) # set resolution of 20 µm for the profile
abundance.show_profile('auto') # plot profile (Fig.5.6(b))
