# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:31:03 2022

@author: juand
"""

from spectramap import spmap as sp

plastics = sp.hyper_object('plastics')
plastics.read_csv_xz('layers')

plastics.show(True)

plastics.keep(400, 1850) # keeping finger print and high wavenumber region
plastics.gaussian(2) # appliying gaussian filter
plastics.rubber() # rubber baseline correction
plastics.vector()

plastics.kmeans(6) # kmeans clustering example for main_label
main_label = plastics.get_label() # saving the main_label
main_label.name = "main_label" # renaming the title of the label
plastics.show_stack(0,0, "auto") # showing the 6 components

scores_pca, loadings_pca = plastics.pca(3, False) # 3 components pca
scores_pca.show_scatter(main_label, 15, "auto") # showing scatter with sublabel

scores_pls, loadings_pls = plastics.plslda(3, 1) # 3 components pls-lda  and 70% training data
scores_pls.show_scatter(main_label, 15, "auto") # showing scatter with sublevel