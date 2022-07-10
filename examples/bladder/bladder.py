# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 18:02:07 2022

@author: juand
"""

from spectramap import spmap as sp
bladder = sp.hyper_object("bladder")
bladder.read_csv_xz("bladder")

bladder.set_resolution(0.3) ## 300 µm step size resolution
bladder.vector() # vector normalization
original = bladder.get_data() ## get data
### K-means clustering
bladder.kmeans(3) #K-means clustering: 3 components
bladder.remove_label([1])
colors = bladder.show_map(['black', 'green'], None, 1) # 
bladder.show_stack(0, 0, colors) #show stack of the clusters (Fig. 5.5(c))