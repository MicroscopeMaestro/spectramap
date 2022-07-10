# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:52:28 2022

@author: juand
"""

from spectramap import spmap as sp
### reading ###
path = 'examples/microplastic+tissue/data'
micro = sp.hyper_object('MP') #creating the hyper object Wavenumber (cm-1) No. of pixels Wavenumber (cm-1
micro.read_csv(path) #reading the csv file
### processing ###
micro.dbscan(5, 0.5) # hierarchical density-based clustering
colors = micro.show_map(['gray', 'k', 'r'], None, 1) # 2D map of the clusters(Fig.5.6(a))
micro.show_stack(0,0, colors) # stack of the spectral clusters (Fig. 5.6(b))