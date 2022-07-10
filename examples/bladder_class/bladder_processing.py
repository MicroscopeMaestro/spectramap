# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 18:55:36 2022

@author: juand
"""

from spectramap import spmap as sp

#%% reading csv.xz files
    
bladder = sp.hyper_object("bladder") ## the library
bladder.read_csv_xz("long_bladder")

#%% first sight of the spectra
peaks = bladder.show2(True, 0.2, "r") # basic plotting

#%% processing data

bladder.rubber() # rubber band baseline (basic)
bladder.gol(11, 3, 0) #savitzky golay filter

bladder.remove_label(['Lipid+Protein', 'Protein+Lipid', 'Tumor+Lipid']) ## remove lipids
bladder.rename_label(["Tumor+Fibrosis+Protein", "Necrosis+Tumor", "Tumor+Necrosis"], ["Tumor", "Necrosis", "Necrosis"]) ## remove nonclear data
bladder.rename_label(["Tumor", "Necrosis", "Protein", "Lipid"], ["T", "N", "P", "L"]) # rename labels

bladder.show_spectra(0, 0, 'auto')
scores, loadings = bladder.pca(3, False) ## principal component analysis
loadings.show_stack(0.2,1,"auto") ## show stacked plot
scores.show_scatter(bladder.get_label(), 18, "auto") ## show scatter plot

bladder.hca("euclidean", "ward", 0.5, 12) # hierarchical clustering analysis
scores.show_scatter(bladder.get_label(), 18, "auto")
bladder.show_spectra(0,0,"auto")
