## -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:00:58 2022

@author: juand
"""

from spectramap import spmap as sp
### Defining paths ###
blue_path = 'examples/plastics/blue'
red_path = 'examples/plastics/red'
natural_path = 'examples/plastics/natural'

### Creating and reading the files ###
red = sp.hyper_object('red') ## Declaring the first hyper object for red sample
red.read_csv_xz(red_path) ## Reading the comma separated vector file
red.set_label('red') ## Setting red name to the whole data set spectra
#red.keep(500, 1800) ## Keep finger region
meanred = red.mean() ## Compute mean of all red spectra

blue = sp.hyper_object('blue') ## Same procedure for the blue sample
blue.read_csv_xz(blue_path)
blue.set_label('blue')
#blue.keep(500, 1800)
#blue.rubber() ## Baseline correction rubber band
meanblue = blue.mean()
natural = sp.hyper_object('natural') ## Same procedure for the natural sample
natural.read_csv_xz(natural_path)
red.set_label('natural')
#natural.keep(500, 1800)
meannatural = natural.mean()
### Concataneting the three hyperspectral objects
concat = sp.hyper_object('concat') ## create a new empty object
concat.concat([meanred, meanblue, meannatural])
concat.show_stack(0,0,['red', 'blue', 'gray']) ## show the spectra (see Fig. 5.4)
