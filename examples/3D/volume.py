# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:56:52 2022

@author: juand
"""

from spectramap import spmap as sp
import pandas as pd

### reading ###
path = 'examples/3D/cube'
cube = sp.hyper_object('cube') #creating the hyper object

pre_result = pd.read_table(path+'.csv.xz', sep=',')

cube.read_csv_3d_xz(path) #reading the 3d csv file and placing theresolutions xy = 60 µm and z = 70 µm
### preprocessing ###
cube.keep(500, 1800) #finger print selection
cube.airpls(100) #advanced baseline correction
cube.vector() #vector normalization
### processing ###
vca = cube.vca(4) # number of expected components
vca.show_stack(0, 0, 'auto')
abundance = cube.abundance(vca, 'NNLS') # concentration estimation by NNLS
aux = abundance.show_intensity_volume(0.5) # 3d plot of all clusters (Fig. 5.8(b-c))
