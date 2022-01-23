# -*- coding: utf-8 -*-
"""
@author: Juan-David Munoz-Bolanos
@main_contributors: Ecehan Cevik, Dr. Tanveer Shaik, Dr. Christoph Krafft, Shivani Sharma
"""
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
#import hdbscan
from sys import exit
import spc_spectra as spc
from os import listdir
from os.path import isfile, join
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_absolute_error, auc, silhouette_score
from pyspectra.readers.read_spc import read_spc, read_spc_dir
from scipy.spatial import ConvexHull
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.cross_decomposition import PLSRegression
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.patches as mpatches
from math import nan
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.colors as colors_map
from matplotlib.widgets import EllipseSelector
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA 
from scipy.io import loadmat
import matplotlib as mpl
from pylab import arange,pi,sin,cos,sqrt
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
import scipy.linalg as splin
from scipy.ndimage import gaussian_filter1d
import sys
import scipy.linalg as splin
from scipy.stats import pearsonr
import scipy.optimize as opt
import scipy as sp
from sklearn.metrics import r2_score

#############################################
# Internal functions
#############################################

def plot_conditions():
    """
    Standard condition for plotting the plots

    Returns
    -------
    fig_size : float
        the size of the frame of the plot.

    """
    fig_width_pt = 300  # 246 Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height =fig_width*golden_mean       # height in inches
    fig_size = [fig_width,fig_height] 
    
    # Edit the font, font size, and axes width
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['figure.figsize'] = fig_size
    return fig_size
        
def find_pixel(data, lower):
    """
    It finds the pixel location (index) with the wavenumber input

    Parameters
    ----------
    data : float array
        data of the pixels.
    lower : float
        target wavenumber.

    Returns
    -------
    index_lower : float
        index.

    """
    setted_list = pd.to_numeric(data.columns).to_list()
    value_chosen = lower
    minimum = float("inf")
    count = 0
    for val in setted_list:
        count+=1
        if abs(val - value_chosen) < minimum:
            final_value = val
            index_lower = count
            minimum = abs(val - value_chosen)
    return (index_lower)
      
def NNLS(M, U):
    """
    NNLS performs non-negative constrained least squares of each pixel
    in M using the endmember signatures of U.  Non-negative constrained least
    squares with the abundance nonnegative constraint (ANC).
    Utilizes the method of Bro.

    Parameters
    ----------
    M : numpy array
        2D data matrix (N x p).
    U: numpy array
        2D matrix of endmembers (q x p).

    Returns
    -------
    X : array
        An abundance maps (N x q).

    References:
        Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401.
    """
    N, p1 = M.shape
    q, p2 = U.shape
    X = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(U, U.T)
    for n1 in range(N):
        X[n1] = opt.nnls(MtM, np.dot(U, M[n1]))[0]
    return X

def create_empty_hyper_object(data):
    """
    Creates empty hyper object with the input data

    Parameters
    ----------
    data : Pandas DataFrame
        input for the intensities.

    Returns
    -------
    new : hyper object
        empty object with the input data.

    """
    new = hyper_object("new")
    new.set_data(data)
    new.set_label("new")
    new.set_position(pd.DataFrame(np.zeros((len(data), 2))))
    return new

def OLS(M, U):
    """
    NNLS performs linear constrained least squares of each pixel
    in M using the endmember signatures of U.  Non-negative constrained least
    squares with the abundance nonnegative constraint (ANC).
    Utilizes the method of Bro.

    Parameters:
        M: `numpy array`
            2D data matrix (N x p).

        U: `numpy array`
            2D matrix of endmembers (q x p).

    Returns: `numpy array`
        An abundance maps (N x q).

    References:
        Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401.
    """
    N, p1 = M.shape
    q, p2 = U.shape
    X = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(U, U.T)
    for n1 in range(N):
        # opt.nnls() return a tuple, the first element is the result
        X[n1] = opt.lsq_linear(MtM, np.dot(U, M[n1]))[0]
    return X

def read_mat(path):
    return loadmat(path)

def pearson_affinity(M):
   return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])

def estimate_snr(Y,r_m,x):
  [L, N] = Y.shape           # L number of bands (channels), N number of pixels
  [p, N] = x.shape           # p number of endmembers (reduced dimension)
  P_y     = sp.sum(Y**2)/float(N)
  P_x     = sp.sum(x**2)/float(N) + sp.sum(r_m**2)
  snr_est = 10*sp.log10( (P_x - p/L*P_y)/(P_y - P_x) )
  return snr_est


def vca(Y,R,verbose = True,snr_input = 0):
    """
    Vertex Component Analysis
    
    Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    
    ------- Input variables -------------
      Y - matrix with dimensions L(channels) x N(pixels)
          each pixel is a linear mixture of R endmembers
          signatures Y = M x s, where s = gamma x alfa
          gamma is a illumination perturbation factor and
          alfa are the abundance fractions of each endmember.
      R - positive integer number of endmembers in the scene
    
    ------- Output variables -----------
    Ae     - estimated mixing matrix (endmembers signatures)
    indice - pixels that were chosen to be the most pure
    Yp     - Data matrix Y projected.   
    
    ------- Optional parameters---------
    snr_input - (float) signal to noise ratio (dB)
    v         - [True | False]
    ------------------------------------
    """
    #############################################
    # Initializations
    #############################################
    if len(Y.shape)!=2:
      sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')
    [L, N]=Y.shape   # L number of bands (channels), N number of pixels   
    R = int(R)
    if (R<0 or R>L):  
      sys.exit('ENDMEMBER parameter must be integer between 1 and L')   
    #############################################
    # SNR Estimates
    #############################################
    if snr_input==0:
      y_m = sp.mean(Y,axis=1,keepdims=True)
      Y_o = Y - y_m           # data with zero-mean
      Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
      x_p = sp.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace
      SNR = estimate_snr(Y,y_m,x_p);
      if verbose:
        print("SNR estimated = {}[dB]".format(SNR))
    else:
      SNR = snr_input
      if verbose:
        print("input SNR = {}[dB]\n".format(SNR))
    SNR_th = 15 + 10*sp.log10(R)
    #############################################
    # Choosing Projective Projection or 
    #          projection to p-1 subspace
    #############################################
    if SNR < SNR_th:
      if verbose:
        print("... Select proj. to R-1")
        d = R-1
        if snr_input==0: # it means that the projection is already computed
          Ud = Ud[:,:d]
        else:
          y_m = sp.mean(Y,axis=1,keepdims=True)
          Y_o = Y - y_m  # data with zero-mean 
           
          Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:d]  # computes the p-projection matrix 
          x_p =  sp.dot(Ud.T,Y_o)                 # project thezeros mean data onto p-subspace
                  
        Yp =  sp.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L             
        x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = sp.amax(sp.sum(x**2,axis=0))**0.5
        y = sp.vstack(( x, c*sp.ones((1,N)) ))
    else:
      if verbose:
        print("... Select the projective proj.")
      d = R
      Ud  = splin.svd(sp.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
      x_p = sp.dot(Ud.T,Y)
      Yp =  sp.dot(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)           
      x =  sp.dot(Ud.T,Y)
      u = sp.mean(x,axis=1,keepdims=True)        #equivalent to  u = Ud.T * r_m
      y =  x / sp.dot(u.T,x)
    #############################################
    # VCA algorithm
    ############################################# 
    indice = sp.zeros((R),dtype=int)
    A = sp.zeros((R,R))
    A[-1,0] = 1
    for i in range(R):
      w = sp.random.rand(R,1);   
      f = w - sp.dot(A,sp.dot(splin.pinv(A),w))
      f = f / splin.norm(f)      
      v = sp.dot(f.T,y)
      indice[i] = sp.argmax(sp.absolute(v))
      A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))
    Ae = Yp[:,indice]
    return (Ae,indice,Yp)
  
def snip(raman_spectra,niter):
    #snip algorithm
    assert(isinstance(raman_spectra, pd.DataFrame)), 'Input must be pandas DataFrame'
    spectrum_points = len(raman_spectra.columns)
    raman_spectra_transformed = np.log(np.log(np.sqrt(raman_spectra +1)+1)+1)
    working_spectra = np.zeros(raman_spectra.shape)
    for pp in np.arange(1,niter+1):
        r1 = raman_spectra_transformed.iloc[:,pp:spectrum_points-pp]
        r2 = (np.roll(raman_spectra_transformed,-pp,axis=1)[:,pp:spectrum_points-pp] + np.roll(raman_spectra_transformed,pp,axis=1)[:,pp:spectrum_points-pp])/2
        working_spectra = np.minimum(r1,r2)
        raman_spectra_transformed.iloc[:,pp:spectrum_points-pp] = working_spectra

    baseline = (np.exp(np.exp(raman_spectra_transformed)-1)-1)**2 -1
    return baseline

def read_spc(filename):
    ##read scp
    f=spc.File(filename) #Read file
    #Extract X & y
    if f.dat_fmt.endswith('-xy'):
        for s in f.sub:
            x=s.x
            y=s.y
    else:
        for s in f.sub:
            x = f.x
            y = s.y

    Spec=pd.Series(y,index=x)
    return Spec

def rubberband(x, y):
    ###rubber band algorithm
     result = (x, y)
     #re = np.array(result)
     v = ConvexHull(np.array(result).T).vertices
     v = np.roll(v, -v.argmin())
     v = v[:v.argmax()]
     return np.interp(x, x[v], y[v])

def snake_table(numx, numy):
    """
    It creates a snake table

    Parameters
    ----------
    numx : int
        number of cols.
    numy : int
        number of rows.

    Returns
    -------
    table : TYPE
        DESCRIPTION.

    """
    table = pd.DataFrame(np.zeros((numx*numy, 2)))
    table = table.rename(columns = {0:'x', 1:'y'})
    
    count1 = 0
    flag = 0
    for count in range (numx*numy):
        
        if count1 == 0:
            flag = 0
        if count1 == numx:
            flag = 1
        
        if flag == 0:
            
            table.iloc[count,0] = count1
            count1+=1
            
        if flag == 1:
            count1-=1
            table.iloc[count,0] = count1
            
    count1 = 0
    count2 = 0
    for count in range (numx*numy):
        
        if count1 == numx:
            count1 = 0
            count2+=1
        table.iloc[count,1] = count2
        count1+=1
    return table

def plot_dendrogram(model, **kwargs):
    #plot dendogram
    dist = kwargs.pop('max_d', None)
    # Create linkage matrix and then plot the dendrogram
    fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height =fig_width*golden_mean       # height in inches
    fig_size = [fig_width,fig_height] 
    
    # Edit the font, font size, and axes width
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['figure.figsize'] = fig_size
    fig  = plt.figure(figsize = fig_size, dpi = 300)
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(model, **kwargs)
    plt.axhline(y=dist, c='k', linestyle='dashed')
    plt.tight_layout()
    
def fixer(y, m, limit):    
    #fix the spike gap in the spectrum
    spikes = abs(np.array(modified_z_score(np.diff(y)))) > limit
    y_out = y.copy() # So we don’t overwrite y
    #print('length', len(spikes))
    for i in np.arange(m, len(spikes)-m-1):
            if spikes[i] != 0: # If we have an spike in position i
                
                w = np.arange(i-m,i+m+1) # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
                if np.unique(np.isnan(w2)) == True:
                    y_out[i] = y_out[i-1]
                    print('Error due to')
                else:
                    if len(w2)>0:
                        y_out[i] = np.mean(y[w2]) # and we take the median 
                    else:
                        #print('small')
                        #y_out[i] = np.nan
                        y_out[i] = y[i] 
                #print('spike at pixel', i)
    return y_out

def peak_finder(num, axs, normalized_avg, prominence, color, digits):
    #find the peaks positions
    distance  = 10
    expn = normalized_avg.to_numpy()
    exp = normalized_avg/normalized_avg.max()
    exp = exp.to_numpy()
    height = normalized_avg.max()-normalized_avg.max()*0.2
    wave =  pd.to_numeric(normalized_avg.index).to_numpy()
    peaks, _ = find_peaks(exp, prominence = prominence, distance = distance)
    index = peaks.copy()
    for count in range(len(peaks)):
        value_chosen = peaks[count]
        minimum = float("inf")
        count1 = 0
        for value in wave:
            count1+=1
            if abs(value - value_chosen) < minimum:
                index[count] = count1
                minimum = abs(value - value_chosen)
    keep = []    
    for item in peaks:
        
        if digits == 0:
            axs.annotate(int(wave[item]), xy = (wave[item]+5, height), rotation = 90, size = 8, color = color)
            axs.axvline(x = int(wave[item]), linestyle='--', linewidth = 0.6, alpha = 0.5, color = color)
            keep.append(int(wave[item]))
        else:
            axs.annotate(np.round(wave[item],digits), xy = (wave[item]+5, height), rotation = 90, size = 8, color = color)
            axs.axvline(x = wave[item], linestyle='--', linewidth = 0.6, alpha = 0.5, color = color)
            keep.append(np.round(wave[item] ,digits))
    return keep

def modified_z_score(delta_int):
    #spikes removing algorithm
    median_int = np.median(delta_int)
    mad_int = np.median([np.abs(delta_int-median_int)])
    modified_z_scores = 0.6745*(delta_int-median_int)/mad_int        
    return modified_z_scores

def save_data(path, data, name):
    #saving hyperspectral data in csv
    gfg_csv_data = pd.DataFrame(data).to_csv(path + name + '.csv', index = False, header = True) 

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_, porder = 1, itermax = 50):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print( 'WARNING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z
     
########################################
##### definition of hyper_object #######
########################################
   
class hyper_object:
    def __init__(self, name):
        """
        The starter 

        Parameters
        ----------
        name : (string)
            Name of the hyper object.

        Returns
        -------
        None.

        """
        self.data = pd.DataFrame()    
        self.position = pd.DataFrame()
        self.name = name
        self.resolution = 1
        self.original = pd.DataFrame()
        self.n = 0
        self.m = 0
        self.resolutionz = 1
        self.label = pd.DataFrame()
        
    def get_data(self):
        """
        it returns a copy of the intensity dataframe 

        Returns
        -------
        Pandas DataFrame

        """
        return (self.data.copy())
    
    def get_position(self):
        """
        It returns the copy of the pixel positions

        Returns
        -------
        Pandas DataFrame

        """
        return (self.position.copy())
    
    def set_name(self, name):
        """
        Set the name of the hyper object

        Parameters
        ----------
        name : strig
            name.

        Returns
        -------
        None.

        """
        self.name = name
    
    def set_data(self, data):
        """
        It sets the data intensity dataframe

        Parameters
        ----------
        data : Pandas DataFrame
            Intensity Frame.

        Returns
        -------
        None

        """
        self.data = pd.DataFrame(data).dropna()

    def set_original(self, data):
        self.original = data
    
    def set_position(self, position):
        """
        It sets the 2D dataframe position

        Parameters
        ----------
        position : float Pandas DataFrame
            x and y pixel positions.

        Returns
        -------
        None.

        """
        self.position = position.copy()
        self.position.columns = ['x', 'y']
        self.m = int(self.position['x'].max() + 1)
        self.n = int(self.position['y'].max() + 1)
        
    def set_position_3d(self, position):
        """
        It sets the 3D dataframe positions

        Parameters
        ----------
        position : DataFrame
            pixel position in 3D.

        Returns
        -------
        None.

        """
        self.position = position
        self.m = int(position['x'].max()) + 1
        self.n = int(position['y'].max()) + 1
        self.l = int(position['z'].max()) + 1
        
    def rename_label(self, before, after):
        """
        Rename the labels

        Parameters
        ----------
        before : list
            current list labels.
        after : list
            to new name labels.

        Returns
        -------
        None.

        """
        self.reset_index()
        for count in range (len(before)):
            self.label[self.label.iloc[:] == before[count]] = after[count]
        
        concat = pd.concat([self.data, self.position, self.label], axis = 1).dropna()
        self.label = concat['label']
        self.position = concat[['x', 'y']]
        self.data = concat.iloc[:, :len(self.data.columns)]
        
    def diff(self, hyper_object):
        """
        Substraction betwen base hyper_object and hyper_object (single spectrum)

        Parameters
        ----------
        hyper_object : hyperobject
            substraction between hyperobject and base hyperobject.

        Returns
        -------
        hyper_object.
            result of the substraction
        """
        copy = self.copy()
        if len(self.data) > 1:
            for count in range(len(self.data)):
                copy.data.iloc[count, :] = self.data.iloc[count, :].subtract(hyper_object.get_data())
        else:
            copy.data = self.data - hyper_object.data
        return copy
    
    def show_scatter(self, colors, label, sub_label, size):
        """
        It plots the scatter plot of the 2 first components 

        Parameters
        ----------
        colors : 'auto' or string list
            colors for the plot.
        label : Series
            array of label legend
        size : float
            size of the scattered points.
        sublabel : Series or empty list
            array of sublabel legend
        Returns
        -------
        None.

        """
        print('2D scatter')
        fig_size = plot_conditions()
        fig, ax = plt.subplots(figsize = fig_size, dpi = 300)

        plt.xlabel('component 1')
        plt.ylabel('component 2')
        x= self.data.iloc[0, :].values
        y= self.data.iloc[1, :].values
        
        sub_label_exist = True
        if len(sub_label) == 0:    
            sub_label_exist = False
        
        if sub_label_exist == True:
            markers = ["o", "v", "X", "^", "s", "D", "<"]
            #markers = ['o', 'o', 'o', 'o', 'o', "o", "o", "o"]
            sub_unique = sub_label.unique()
            #sub_label_string = np.zeros((len(x), len(y)))
            sub_label_string = np.ones(len(x), dtype = str)
            aux = 0
            #print(sub_label_string)
            for count in range(len(x)):
                for count2 in range(len(sub_unique)):
                    if sub_unique[count2] == sub_label.iloc[count]:
                        sub_label_string[count] = markers[count2]   
        #return(sub_label_string)
        if self.label.empty == 1:
            x= self.data.iloc[0, :].values
            y= self.data.iloc[1, :].values
            ax.scatter(x, y, s = size, marker = 'o', alpha = 0.7, edgecolor = 'k', linewidths = 0.1)
        else:         
            unique = label.unique()
            c = label
            length = len(unique)
            print('Num of labels:', length)
            if length > 1:
                if colors == 'auto':
                    colormap = cm.get_cmap('hsv')
                    norm = colors_map.Normalize(vmin=0, vmax=length)
                    colors = colormap(norm(range(length)))
                newcolors = np.ones(len(label), dtype = object)
                newcolors[:] = 'white'
                for count in range(len(label)):
                    for count1 in range (len(unique)):
                        if label.iloc[count] == unique[count1]:
                            newcolors[label.index[count]] = colors[count1]
                c = newcolors 
        
            if length > 1:
                if sub_label_exist == False: 
                    ax.scatter(x, y, c = c, s = size, marker = 'o', alpha = 0.7, edgecolor = 'k', linewidths = 0.1)
                    patch = []
                    for count in range(len(unique)):
                        patch.append(plt.Line2D([],[], marker="o", ms=4, ls="", mec=None, color=colors[count], label=unique[count]))
                               
                    ax.legend(handles=patch, loc = 2,  bbox_to_anchor=(0.97,1), borderaxespad=0, frameon = False)
                else:
                    for xp, yp, cp, m in zip(x, y, c, sub_label_string):
                        ax.scatter(xp, yp, c = colors_map.to_hex(cp), s = size, marker = m, alpha = 0.7, edgecolor = 'k', linewidths = 0.1)
                    patch = []
                    for count in range(len(unique)):
                        patch.append(plt.Line2D([],[], marker="o", ms=4, ls="", mec=None, color=colors[count], label=unique[count]))
                    legend1 = ax.legend(handles=patch, loc = 2,  bbox_to_anchor=(0.97,1.1), borderaxespad=0, frameon = False, title = label.name)
                    patch = []
                    for count in range(len(sub_unique)):
                        patch.append(plt.Line2D([],[], marker=markers[count], ms=4, ls="", mec=None, color="k", label=sub_unique[count]))
                    ax.legend(handles=patch, loc = 2,  bbox_to_anchor=(0.97, 0.3), borderaxespad=0, frameon = False, title = sub_label.name)
                    ax.add_artist(legend1)
            else:
                ax.scatter(x, y, s = size, marker = 'o', alpha = 0.4, edgecolor = 'k', linewidths = 0.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False) 
            plt.tight_layout()
    
    def scatter_3D(self, color):
        """
        Plots the 3D scatter of 3d position

        Parameters
        ----------
        color : 'auto' or string list
            colors for the scatter points.

        Returns
        -------
        None.

        """
        fig_size = plot_conditions() 
        fig = plt.figure(figsize=fig_size, dpi = 300)
        ax2 = fig.add_subplot(111, projection='3d')
        aux = np.zeros((self.m, self.n, self.l)).reshape(self.m, self.n, self.l)
        aux[:] = np.nan
        x = self.position['x']
        y = self.position['y']
        z = self.position['z']
        cluster = self.label.copy()
        unique = cluster.unique()
        if len(unique) < 10:
            newcolors = np.ones(len(self.position), dtype = object)
            newcolors[:] = 'white'
            for count in range(len(cluster)):
                for count1 in range (len(unique)):
                    if cluster.iloc[count] == unique[count1]:
                        newcolors[cluster.index[count]] = color[count1]
            c = newcolors
        else:
            c = cluster
        p = ax2.scatter3D(x*self.resolution, y*self.resolution, z*self.resolutionz, c=c, s = 50, alpha = 0.5, linewidths=0.1)
        ax2.set_zlabel('$Z [mm]$')
        ax2.set_xlabel('$X [mm]$')
        ax2.set_ylabel('$Y [mm]$')
        plt.show()
    
    def clean_position(self):
        """
        Cleans the size defects of position

        Returns
        -------
        None.

        """
        self.position = self.position.iloc[self.data.index, :]
        #
    def interpolate(self, ratio):
        """
        Interpolation through wavenumber

        Parameters
        ----------
        ratio : the proportion to interpolate
            how many times the wavenumber expands (2, 3, 4, ...).

        Returns
        -------
        None.

        """
        x = self.data.columns.to_numpy()
        y = self.data.to_numpy()
        xnew = np.linspace(x.min(), x.max(), len(x)*ratio) 
        y_smooth = np.zeros((len(y), len(xnew)))
        for count in range(len(self.data)):
            spl = make_interp_spline(x, y[count], k=3)
            y_smooth[count] = spl(xnew)
        self.data = pd.DataFrame(y_smooth)
        self.data.columns = xnew
        self.resolution = self.resolution/ratio
            
    def rubber(self):
        """ 
        Compute rubber band correction (good for converting all values to possitive)
        
        Returns
        -------
        None.

        """
        aux = self.data.copy()
        x = pd.to_numeric(aux.columns).values
        for count1 in range(len(aux)):
            y = aux.iloc[count1,:].values
            baseline = rubberband(x, y)
            aux.iloc[count1,:] = aux.iloc[count1,:].values - baseline
            print(count1+1, '/', len(aux))
        self.data = aux
        print('Done')
        
    def copy(self):
        """
        Copy of the hyperobject

        Returns
        -------
        new : hyperobject
            copied hyperobject.

        """
        new = hyper_object(self.name + '_copy')
        new.set_data(self.data)
        if len(self.position.columns) ==2:
            new.set_position(self.position)
        else:
            new.set_position_3D(self.position)
            new.l = self.l
            new.resolutionz = self.resolutionz

        new.set_resolution(self.resolution)
        new.set_label(self.label)
        new.m = self.m
        new.n = self.n
        return new
    
    def get_intensity(self, wave):
        """
        it gets the intensity at certain wavenumber position

        Parameters
        ----------
        wave : float
            wavenumber.

        Returns
        -------
        intensity Hyper_object
            the data dataframe contains the intensity at the peak.

        """
        index_lower = find_pixel(self.data, wave)
        abu = hyper_object(self.name + '_intensity')
        abu.data = pd.DataFrame(self.data.iloc[:, index_lower])
        try:
            abu.set_position(self.position)
        except:
            print('No Coordinates')
        abu.set_label([wave])
        abu.set_resolution(self.resolution)
        abu.m = self.m
        abu.n = self.n
        return abu
    
    def get_area(self, lower, upper):
        """
        Calculate the area under the curve
        
        Parameters
        ----------
        lower : float
            lower wavenumber axis.
        upper : float
            upper wavenumber axis.

        Returns
        -------
        abu : intensity hyper_object
            area under the curve.

        """
        lower_wave = find_pixel(self.data, lower)
        upper_wave = find_pixel(self.data, upper)
        area = pd.Series(np.zeros((len(self.data.index))))
        ranging = self.data.iloc[:, lower_wave:upper_wave].copy()
        for count in range(len(ranging.index)):
            selection = ranging.iloc[count, :]
            area.iloc[count] = np.trapz(selection, pd.to_numeric(ranging.columns))
        abu = hyper_object(self.name + '_area')        
        try:
            abu.set_position(self.position)
        except:
            print('No Coordinates')
        #abu.set_label('area')
        abu.data = pd.DataFrame(area)
        abu.set_resolution(self.resolution)
        abu.m = self.m
        abu.n = self.n
        return abu
    
    def get_intensity_3D(self, wave):
        """
        it gets the intensity at certain wavenumber position

        Parameters
        ----------
        wave : float
            wavenumber.

        Returns
        -------
        Series
            the intensity at the wavenumber position of whole dataset.

        """
        index_lower = find_pixel(self.data, wave)        
        return self.data.iloc[:, index_lower]
    
    def show_stack(self, enable, center, colors):
        """
        It plots the labeled spectra in a stack shape with standard deviation

        Parameters
        ----------
        enable : float
            0: no finding peaks if > 0 activaiton of peak labeling by tuning the prominence.
        center : float
            0 draw of center line and other number creates an offset of the labels.
        colors : list of strings or 'auto'
            colors for the spectra visualization.

        Returns
        -------
        None.

        """
        path = None
        type_file = 'png'
        values = self.label.unique()
        indices = []
        final = []
        fig_size = plot_conditions()
        fig, axs = plt.subplots(len(values), sharex = 'all', sharey = 'all', figsize = fig_size, dpi = 300, gridspec_kw = {'left': 0.05, 'bottom':0.180, 'right':0.95, 'top':0.9, 'wspace':0, 'hspace':0})
        average = pd.DataFrame()
        normalized_avg = pd.DataFrame()
        normalized_std = pd.DataFrame()
        std = pd.DataFrame()
        concat = pd.DataFrame()
        count = 0
        
        if colors == 'auto':
            colormap = cm.get_cmap('hsv')
            norm = colors_map.Normalize(vmin=0, vmax=len(values))
            colors = colormap(norm(range(len(values))))

        if len(values) > 1:
            for count in range(len(values)):
                frame = self.data[self.label == values[count]]
                if len(frame.index) > 1:
                    average = frame.mean()
                    std = frame.std()
                    maximum_avg = average.max()
                    minimum_avg = average.min()
                    normalized_avg = (average - minimum_avg) / (maximum_avg - minimum_avg)
                    normalized_std = (std - minimum_avg) / (maximum_avg - minimum_avg)
                    axs[count].plot(pd.to_numeric(self.data.columns), normalized_avg, label = values[count], linewidth = 0.7, color = colors[count])
                    axs[count].legend([values[count]], frameon = False, loc = 'upper left')
                    axs[count].plot(pd.to_numeric(frame.columns), normalized_avg - normalized_std, color = colors[count], linewidth = 0.3)
                    axs[count].plot(pd.to_numeric(frame.columns), normalized_avg + normalized_std, color = colors[count], linewidth = 0.3)
                    axs[count].fill_between(pd.to_numeric(frame.columns), normalized_avg - normalized_std, normalized_avg + normalized_std, alpha = 0.2, color = colors[count])
                    axs[count].xaxis.set_major_locator(mpl.ticker.MultipleLocator(300))
                    axs[count].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
                    axs[count].spines['top'].set_visible(False)
                    axs[count].spines['bottom'].set_visible(False)
                    axs[count].spines['left'].set_visible(False)
                    axs[count].spines['right'].set_visible(False)
                    axs[count].tick_params(left = False)
                    axs[count].xaxis.set_visible(True)
                    axs[count].set(yticklabels=[])  
                    leg = axs[count].legend(frameon = False, loc = 'upper left', bbox_to_anchor=(0, 1), handlelength=0.4)
                    for line in leg.get_lines():
                        line.set_linewidth(4)
                else:  
                    average = frame.mean()
                    maximum_avg = average.max()
                    minimum_avg = average.min()           
                    normalized_avg = (average - minimum_avg) / (maximum_avg - minimum_avg) 
                    axs[count].plot(pd.to_numeric(frame.columns), normalized_avg, label = values[count], linewidth = 0.7, color = colors[count])
                    axs[count].legend([values[count]], frameon = False, loc = 'upper left')
                    axs[count].xaxis.set_major_locator(mpl.ticker.MultipleLocator(300))
                    axs[count].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
                    axs[count].spines['top'].set_visible(False)
                    axs[count].spines['right'].set_visible(False)
                    axs[count].spines['bottom'].set_visible(False)
                    axs[count].spines['left'].set_visible(False)
                    axs[count].tick_params(left = False)
                    axs[count].xaxis.set_visible(False)
                    axs[count].set(yticklabels=[])  
                    leg = axs[count].legend(frameon = False, loc = 'upper left', bbox_to_anchor=(0, 1), handlelength=0.4)
                    for line in leg.get_lines():
                        line.set_linewidth(4)
                stick = pd.DataFrame(average).T
                stick['label'] = values[count]
                concat = pd.concat([concat, stick])
                
                if center == 1:
                    axs[count].plot(pd.to_numeric(frame.columns), np.zeros(len(frame.columns))+0.5, linewidth = 0.4, color = 'grey', alpha = 0.7)

                if enable > 0:
                    peak_finder(count, axs[count], normalized_avg, enable, colors[count], 0)
    
        axs[count].xaxis.set_visible(True)
        axs[count].spines['bottom'].set_visible(True)
        axs[count].set_xlabel('Wavenumber (cm$^{-1}$)')
        plt.subplots_adjust()
        
    def read_single_spc(self, path):
        """
        Reading a single spc file

        Parameters
        ----------
        path : string
            path of the file directory.

        Returns
        -------
        None.

        """
        self.__init__(self.name)
        spec = pd.DataFrame(read_spc(path + '.spc'))
        spec.index = np.round(spec.index.to_numpy(), 2)
        self.label = pd.Series([self.name], name = 'label')
        self.data = self.original = spec.T
        self.position['x'] = 0
        self.position['y'] = 0
        self.m = 1
        self.n = 1
        self.l = 1
        print('Done')

    def add_peaks(self, prominence, color):
        """
        After hyper_object.show() it is possible to add peaks

        Parameters
        ----------
        prominence : float or list of float peaks
            if it is float (how strong the peak finding is or the manual selection of peaks for plotting);
            if it is a list (it defines the peak positions for plotting manually)
        color : string color
            color for plotting.
        Returns
        -------
        None.

        """
        peaks = prominence
        axs = plt.subplot(111)
        if type(peaks) == float:
            high = peak_finder(0, axs, self.data.mean(), peaks, color, 1)
            return (high)
        else:
            offset = 10
            index = peaks
            wave =  pd.to_numeric(self.data.columns).to_numpy()
            expn = self.data.values.max()
            print(expn)
            for count in range(len(peaks)):
                value_chosen = peaks[count]
                minimum = float("inf")
                count1 = 0
                for value in wave:
                    count1+=1
                    if abs(value - value_chosen) < minimum:
                        index[count] = count1
                        minimum = abs(value - value_chosen)
            for item in peaks:
                axs.annotate(int(wave[item]), xy = (np.round(wave[item], 2)+offset, expn), rotation = 90, size = 8, color = color)
                axs.axvline(x = wave[item], color=color, linestyle='--', linewidth = 0.6, alpha = 0.5)
            
    def read_multi_spc(self, path):
        """
        Reading several spc files in the same path directory

        Parameters
        ----------
        path : string
            file directory that contains the spc files.

        Returns
        -------
        None.

        """
        Directory = path
        ext='.spc'
        orient='Row'
        
        self.__init__(self.name)
        #Read all files from directory and create a list, also ensures that the extension of file is correct
        Flist = [f for f in listdir(Directory) if isfile(join(Directory, f)) and f.endswith(ext)]
    
        SpectraDict={}
        #Read all of them an add them to a dictionary
        count = 0
        length = len(Flist)
        for file in Flist:
            Spec=read_spc(Directory + "/" + file)
            Spec.index = np.round(Spec.index.to_numpy(), 2)
            SpectraDict[file]=Spec
            count+=1
            print(count, '/', length)
        #Decide the orientation of dataframe, column-wise or row-wise.
        if orient=='Row':
            SpectraDataFrame=pd.DataFrame(SpectraDict).transpose()
        else:
            SpectraDataFrame = pd.DataFrame(SpectraDict)
        
        df_spc = SpectraDataFrame
        dict_spc = SpectraDict
        self.label = pd.Series(df_spc.index)
        self.data = self.original = df_spc.reset_index(drop = True)
        self.position['x'] = np.arange(len(self.data.index))
        self.position['y'] = np.zeros(len(self.data.index))
        self.m = len(self.data.index)
        self.n = 1
        print('Done')       

    def read_csv_xz(self, file_path):
        """
        Reading a csv.xz file provied it has the standard dataframe structure

        Parameters
        ----------
        file_path : string
            file directory

        Returns
        -------
        None.

        """
        resolution = 1
        file_path = file_path + '.csv.xz'
        pre_result = pd.read_table(file_path, sep=',')
        #pre_result = pre_result.drop(columns = 'Unnamed: 0')
        pre_result = pre_result.dropna(axis = 'rows')
        self.label = pd.Series(pre_result['label'])
        pre_result = pre_result.drop(columns = 'label')
        self.position = pre_result[['x', 'y']]
        pre_result = pre_result.drop(columns = ['x', 'y'])
        self.data = pre_result
        self.position.index = self.data.index
        self.original = self.data
        self.resolution = resolution   
        max_m = pd.to_numeric(self.position['x'])
        self.m = int(max_m.max() + 1)
        max_n = pd.to_numeric(self.position['y'])
        self.n = int(max_n.max() + 1)
        self.data.columns = np.round(pd.to_numeric(self.data.columns), 2)        
        print('Done')
    
   
    def read_csv(self, file_path):
        """
        Reading a csv file provied it has the standard dataframe structure

        Parameters
        ----------
        file_path : string
            file directory

        Returns
        -------
        None.

        """
        resolution = 1
        file_path = file_path + '.csv'
        pre_result = pd.read_table(file_path, sep=',');
        pre_result = pre_result.dropna(axis = 'rows')
        self.label = pd.Series(pre_result['label'])
        pre_result = pre_result.drop(columns = 'label')
        self.position = pre_result[['x', 'y']]
        pre_result = pre_result.drop(columns = ['x', 'y'])
        self.data = pre_result
        self.position.index = self.data.index
        self.original = self.data
        self.resolution = resolution   
        max_m = pd.to_numeric(self.position['x'])
        self.m = int(max_m.max() + 1)
        max_n = pd.to_numeric(self.position['y'])
        self.n = int(max_n.max() + 1)
        self.data.columns = np.round(pd.to_numeric(self.data.columns), 2)        
        print('Done')
        
    def snip(self, iterations):
        """
        Snip baseline correction

        Parameters
        ----------
        iterations : int
            number of interations for the correction.

        Returns
        -------
        None.

        """
        baseline = snip(self.data, iterations)
        self.data = self.data - baseline
        print('Done')
        
           
    def read_csv_3D(self, file_path, x_resolution, z_resolution):
        """
        read a 3D csv data

        Parameters
        ----------
        file_path : string
            path where the file is.
        x_resolution : float
            xy motor stage resolution.
        z_resolution : float
            z motor state resolution.

        Returns
        -------
        None.

        """
        pre_result = pd.read_table(file_path + '.csv', sep=',')
        self.label = pd.Series(pre_result['label'])
        pre_result = pre_result.drop(columns = 'label')
        self.position = pre_result[['x', 'y', 'z']]
        pre_result = pre_result.drop(columns = ['x', 'y', 'z']) 
        self.data = pre_result  
        self.resolution = x_resolution
        self.resolutionz = z_resolution
        max_m = pd.to_numeric(self.position['x'])
        self.m = int(max_m.max() + 1)
        max_n = pd.to_numeric(self.position['y'])
        self.n = int(max_n.max() + 1)
        max_l = pd.to_numeric(self.position['z'])
        self.l = int(max_l.max() + 1)
        self.data = np.round(pd.to_numeric(self.data.columns), 2)
        
    def read_point_1064(self, path_file):
        """
        reads the 1064 txt file if whole files are correct, data, dark and calibration
    
        Parameters
        ----------
        path_file : string
            path file of the files.
        path_calibration : string
            path file of the csv calibraiton file.
    
        Returns
        -------
        None.
    
        """
        data_raw = pd.read_table(path_file + 'data.txt', sep='\t', lineterminator='\n', header = None, usecols = range(512))#, skiprows=[0]);
        calibration_raw = pd.read_table(path_file + 'calibration.csv', sep=',', usecols = range(512))
        #Correction data
        if data_raw.iloc[0,0] == 1 or data_raw.iloc[0,0] == 1:
            data_raw = data_raw.drop(index = 0)
    
        mean_data = data_raw.mean()
        
        dark = pd.read_table(path_file + 'dark.txt', sep='\t', lineterminator='\n', header = None, usecols = range(512));
        if dark.iloc[0,0] == 1 or dark.iloc[0,0] == 1:
            dark = dark.drop(index = 0)
              
        pre_result = mean_data.subtract(dark.mean())
        self.data = pd.DataFrame(pre_result).T
        self.data.columns = calibration_raw.columns
        self.position = pd.DataFrame(np.zeros((1,2)))
        self.position.columns = ['x', 'y']
        self.original = self.data.copy()          
        self.label = pd.Series([self.name], name = 'label')
    
    def intensity_calibration(self, lamp):
        """
        Intensity calibration consideting a well-known illumination source

        Parameters
        ----------
        lamp : hyper object
            intensity of the lamp (many measurements (>10)).
        Returns
        -------
        None.

        """
        lamp.set_label("lamp")
        inten = lamp.mean()
        mymodel = np.poly1d(np.polyfit(pd.to_numeric(inten.get_wavenumber()).to_list(), inten.get_data().T[0].tolist(), 5))
        yaxis = mymodel(self.get_wavenumber())
        norm_yaxis = yaxis/max(yaxis)
        corrected_intensity = self.get_data().values/norm_yaxis
        self.set_data(corrected_intensity)
        
    def calibration_peaks(self, para, sensitivity):
        """
        Peaks finding 

        Parameters
        ----------
        para : hyper_object
            many paracetaminol measurements (> 10)
        sensitivity : float
            threshold for finding the peaks, the lower the more sensitive
        Returns
        -------
        peaks : list
            peaks.

        """  
        para.set_label("para")
        para_copy = para.copy() 
        para_copy.interpolate(10)
        mean = para_copy.mean()
        mean.show(True)
        pixel_peaks = mean.add_peaks(sensitivity, 'r')
        return pixel_peaks
    
    def calibration_regression(self, peak_pixels):
        """
        Polynomial regression to determine the wavenumber axis calibration. The hyper_object is calibrated.

        Parameters
        ----------
        peak_pixels : list 
            pixel position of the peaks in the sensor.

        Returns
        -------
        None.

        """
        real_peaks = [329.2, 390.9, 465.1, 504.0, 651.6, 710.8, 797.2, 834.5, 857.9, 968.7, 1105.5, 1168.5, 1236.8, 1278.5, 1323.9, 1371.5, 1515.1, 1561.6, 1648.4]
        input_string = input("Input the peak that does not match with the pixel_peaks separeted by space:")
        print("\n")
        user_list = input_string.split()
        # print list
        print('list: ', user_list)
        # convert each item to int type
        for i in range(len(user_list)):
            # convert each item to int type
            user_list[i] = float(user_list[i])
            try:
                real_peaks.remove(user_list[i])
            except:
                print("Nonmatching:", user_list[i])
        
        input_string = input("Input the pixel peaks that does not match with the real peaks separeted by space:")
        print("\n")
        user_list = input_string.split()
        # print list
        print('list: ', user_list)
        # convert each item to int type
        for i in range(len(user_list)):
            # convert each item to int type
            user_list[i] = float(user_list[i])
            try:
                peak_pixels.remove(user_list[i])
            except:
                print("Nonmatching:", user_list[i])
                
        mymodel = np.poly1d(np.polyfit(peak_pixels, real_peaks, 3))
        calibration = mymodel(self.get_wavenumber())
        myline = np.linspace(0, max(peak_pixels), 200)
        fig_size = plot_conditions()
        fig = plt.figure(figsize = fig_size, dpi = 300)
        plt.scatter(peak_pixels, real_peaks)
        plt.xlabel("Pixels")
        plt.ylabel("Wavenumber (1/cm)")
        plt.plot(myline, mymodel(myline))
        plt.tight_layout()
        plt.show()
        print("Calibration Error (R): ", r2_score(real_peaks, mymodel(peak_pixels)))
        #return calibration
        self.set_wavenumber(pd.Series(calibration))

    def read_map_1064(self, path_file, resolution):
        """
        reads the 1064 txt file if whole files are correct, data, dark, pb and pbd
    
        Parameters
        ----------
        path_file : string
            path file of the files.
        path_calibration : string
            path file of the csv calibraiton file.
        resolution : float
            xy motor stage resolution.
        Returns
        -------
        None.
        """
        data_raw = pd.read_table(path_file + 'map.txt', sep='\t', lineterminator='\n', header = None, usecols = range(514))#, skiprows=[0]);
        calibration_raw = pd.read_table(path_file + 'calibration.csv', sep=',', skiprows= 3, usecols = range(514))
    
        #Correction data
        if data_raw.iloc[0,0] == 1 or data_raw.iloc[0,0] == 1:
            data_raw = data_raw.drop(index = 0)
        
        dark = pd.read_table(path_file + 'dark.txt', sep='\t', lineterminator='\n', header = None, usecols = range(512));
        if dark.iloc[0,0] == 1 or dark.iloc[0,0] == 1:
            dark = dark.drop(index = 0)
        mean = dark.mean()
        data = data_raw.iloc[:, :512]
        pos = data_raw.iloc[:, 512:514]
        calibration_data = calibration_raw.iloc[:, :512] 
        calibration_pos = calibration_raw.iloc[:, 512:514]
                 
        #Substraction 
        pre_result = data.subtract(dark.mean())
        pre_result.columns = calibration_data.columns   
        pos.columns = ['x', 'y']
        pos.index = pre_result.index
        pos['y'] = pos.iloc[::-1,1].values
        count = 0
        for line in pos['y']:
            line = str(line).rstrip()
            pos.iloc[count, 1] = line
            count+=1
        pos[:] = pos[:].astype(int)
        self.data = pre_result.dropna(True).reset_index(drop = True) 
        self.position = pos
        self.original = self.data.copy()
        self.resolution = resolution
        max_m = pd.to_numeric(self.position['x'])
        self.m = int(max_m.max() + 1)
        max_n = pd.to_numeric(self.position['y'])
        self.n = int(max_n.max() + 1)
        print('Warning: Define label names')
    
    def read_map2_1064(self, path_file, path_calibration, resolution):
        """
        reads the 1064 txt file if whole files are correct, data, dark, pb and pbd
    
        Parameters
        ----------
        path_file : string
            path file of the files.
        path_calibration : string
            path file of the csv calibraiton file.
        resolution : float
            xy motor stage resolution.
    
        Returns
        -------
        None.
    
        """
        data_raw = pd.read_table(path_file + 'data.txt', sep='\t', lineterminator='\n', header = None, usecols = range(514))#, skiprows=[0]);
        calibration_raw = pd.read_table(path_calibration, sep=',', skiprows= 3, usecols = range(514))
    
        #Correction data
        if data_raw.iloc[0,0] == 1 or data_raw.iloc[0,0] == 1:
            data_raw = data_raw.drop(index = 0)
        
        dark = pd.read_table(path_file + 'dark.txt', sep='\t', lineterminator='\n', header = None, usecols = range(512), skiprows=[0,1]);
        mean = dark.mean()
        data = data_raw.iloc[:, :512]
        pos = data_raw.iloc[:, 512:514]
        calibration_data = calibration_raw.iloc[:, :512] 
        calibration_pos = calibration_raw.iloc[:, 512:514]
        
        pb = pd.read_table( path_file + 'pb.txt', sep='\t', lineterminator='\n', header = None, usecols = range(512), skiprows=[0,1]);
        pbd = pd.read_table( path_file + 'pbd.txt', sep='\t', lineterminator='\n', header = None, usecols = range(512), skiprows=[0,1]);
         
        #Substraction 
        diff = data.subtract(dark.mean())
        diffp = pb.mean().subtract(pbd.mean())
        pre_result = pd.DataFrame(diff.values - diffp.values)
        pre_result.columns = calibration_data.columns   
        pos.columns = ['x', 'y']
        pos.index = pre_result.index
           
        pos['y'] = pos.iloc[::-1,1].values
    
        count = 0
        for line in pos['y']:
            line = str(line).rstrip()
            pos.iloc[count, 1] = line
            count+=1
        pos[:] = pos[:].astype(int)
        self.data = pre_result.dropna(True).reset_index(drop = True)
        self.position = pos
        self.original = self.data.copy()
        self.resolution = resolution
        max_m = pd.to_numeric(self.position['x'])
        self.m = int(max_m.max() + 1)
        max_n = pd.to_numeric(self.position['y'])
        self.n = int(max_n.max() + 1)
        self.label = pd.Series(np.arange(len(self.data)))
        self.reset_index()
        
    def gol(self, window, polynomial, order):
        """ Set the value window, polynomial, order
        
        The savitky-golay filter is applied to the full data set
    
        Parameters
        ----------
        window : int
            size of number of neighbor pixels to consider.
        polynomial : int
            order for interpolation.
        order : int
            order of derivation.

        Returns
        -------
        None.

        """
        pre_result = pd.DataFrame(scipy.signal.savgol_filter(self.data, window, polynomial, order, 1))
        pre_result.columns = self.data.columns
        pre_result.index = self.data.index
        self.data = pre_result
    
    def dbscan(self, min_samples, eps):
        """
        It computes the density-based clustering algoirthm

        Parameters
        ----------
        min_samples : int
            minimum number of points to cluster.
        eps : float
            radious of the points.

        Returns
        -------
        None.

        """
        clustering = DBSCAN(min_samples = min_samples, algorithm = 'brute').fit(self.data)
        labels = clustering.labels_
        labels = pd.Series(labels)
        labels.index = self.data.index
        print(labels.unique())
        self.set_label(labels)

    # def hdbscan(self, min_samples, min_cluster):
    #     """
    #     It computes the hierchical density based clusteinrg alogrithm

    #     Parameters
    #     ----------
    #     min_samples or number of neighbors: int
    #         density for the clustreing.
    #     min_cluster : int
    #         minimum cluster .

    #     Returns
    #     -------
    #     None.

    #     """
    #     if min_cluster == 'auto':
    #         clusterer = hdbscan.HDBSCAN(min_samples = min_samples)
    #     else:
    #         clusterer = hdbscan.HDBSCAN(min_samples = min_samples, min_cluster_size = min_cluster)
    #     cluster_labels = clusterer.fit_predict(self.data)
    #     self.set_label(cluster_labels)
    #     print(self.label.unique())
    #     print('Number of clusters: ', len(self.label.unique()))
    #     clusterer.condensed_tree_.plot(select_clusters=True)
    #     return clusterer
        
    def gaussian(self, sigma):
        """ Set the value of sigma
        
        The function applies gaussian smoothing.

        Parameters
        ----------
        sigma : float
            variance of the gaussian filter (how strong the filter is).

        Returns
        -------
        None.

        """
        pre_result = pd.DataFrame(scipy.ndimage.gaussian_filter1d(self.data.copy(), sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0))
        pre_result.columns = self.data.columns
        self.data = pre_result
        
    def remove(self, lower, upper):
        """
        It removes a region in the wavenumber

        Parameters
        ----------
        lower : float
            lower wavenumber.
        upper : float
            upper wavenumber.

        Returns
        -------
        None.

        """
        index_lower = find_pixel(self.data, lower)
        index_upper = find_pixel(self.data, upper)
        aux1 = self.data.iloc[:, :index_lower].copy()
        aux2 = self.data.iloc[:, index_upper:].copy()
        self.data = pd.concat([aux1, aux2], axis = 1)
        
    def keep(self, lower, upper):
        """
        It keeps a region between lower and upper wavenumber

        Parameters
        ----------
        lower : float
            lower wavenumber.
        upper : float
            upper wavenumber.

        Returns
        -------
        None.

        """
        index_lower = find_pixel(self.data, lower)
        index_upper = find_pixel(self.data, upper)
        self.data = self.data.iloc[:, index_lower:index_upper]
                 
    def airpls(self, landa):
        """  
        calculate advanced baseline correction airpls

        Parameters
        ----------
        value : float 
            value represents how strong is the fitting.

        Returns
        -------
        None.

        """
        #Baseline correction
        length = len(self.data.index)
        matrix = np.zeros((length, len(self.data.columns)))
        for item in range(length):
            matrix[item] = airPLS(self.data.iloc[item, :].copy(), landa)
            print('progress : ', item + 1, '/', length)
                
        correction = pd.DataFrame(self.data.to_numpy() - matrix)
        correction.index = self.data.index
        correction.columns = self.data.columns
        self.data = correction
        print('Done')
        
    def read_spc_holo(self, path):
        """
        It reads a spc map from holomaps Kaiser instruments

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        try:
            f = open(path + 'scandata.txt')
    
        except:
            f = open(path + 'Scandata.txt')
            
        read = f.readlines()
        file = []
        follow= 9
        line = read[4]
        res = read[5]
        names = read[12]
        
        test=line[9:len(line)-1]
        test_list = test.split(',')
        numbers = [int(x.strip()) for x in test_list]
        
        x = numbers[1]
        y = numbers[2]
        
        test=res[10:len(res)-1]
        test_list = test.split(',')
        numbers = [float(x.strip()) for x in test_list]
        self.resolution = numbers[0]/1000
    
        name = names.split('\\')
        length = len(name[len(name)-1])
        for count0 in range(int((len(read)-length)/3+1)):
            file.append(read[follow])
            follow+=3
            
        df_spc = pd.DataFrame()
        first = read_spc(path + file[0][len(file[0][:])-length:len(file[0][:])-1]).to_numpy()
        for count in range (0, len(file)):
            try:
                print(count, '/', len(file)+1)
                second = read_spc(path + file[count][len(file[count][:])-length:len(file[count][:])-1]).to_numpy()
                first = np.vstack((first, second))
            except:
                print('not found')
                second = np.zeros(np.shape(second))
                first = np.vstack((first, second))
        raw_data = pd.DataFrame(first)
        raw_data.columns = read_spc(path + file[0][len(file[0][:])-length:len(file[0][:])-1]).index
        table = snake_table(x, y)
        self.position = table
        self.data = raw_data.reset_index(drop = True).copy()
        self.original = raw_data.reset_index(drop = True).copy()
        self.n = y
        self.m = x
        self.label = pd.Series(np.zeros(len(self.data)))
        self.label.rename('label')
        self.data.columns = np.round(self.data.columns, 2)
        print('Done')

    def set_label(self, label):
        """
        Set the label data in hyperobject.label

        Parameters
        ----------
        label : string list (one value or Series) example: ['one'], [1, 2, 3]
            set the label into hyperobject.label.

        Returns
        -------
        None.
        """
        cluster = label
        if len(cluster) > 1:
            if type(cluster) != str:
                cluster = pd.Series(cluster)
                self.label = pd.Series(cluster).dropna()
            else:
                lista = [cluster]*len(self.data)
                self.label = pd.Series(lista)
        else:
            lista = [cluster]*len(self.data)
            self.label = pd.Series( (v[0] for v in lista), name = 'label')
        self.reset_index()
       
    def threshold(self, peak, lower, upper):
        """ 
        Remove the intensiy at peak position out of the lower and upper range
        
        Parameters
        ----------
        peak : float
            peak position.
        lower : float
            lower intensity.
        upper : upper intensity
            DESCRIPTION.

        Returns
        -------
        None.

        """
        index = find_pixel(self.data, peak)
        #index_upper = find_pixel(self.data, upper)
        self.data = self.data[self.data.iloc[:,index] > lower]
        self.data = self.data[self.data.iloc[:,index] < upper]
        self.data = self.data.dropna()
        
    def get_label(self):
        """
        It retuns a copy of label in hyper_object

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.label.copy()
    
    def norm(self):
        """
        It applies a normalization by considering the max value in each spectrum

        Returns
        -------
        None.

        """
        for count in range(len(self.data)):
            self.data.iloc[count, :] = self.data.iloc[count, :]/self.data.iloc[count, :].max()
            
    def norm_at_peak(self, peak):
        """ 
        Normalization considering the peak intensity of each spectrum

        Parameters
        ----------
        peak : float
            peak base for normalization.

        Returns
        -------
        None.

        """
        intensity = self.get_intensity(peak)
        #return intensity
        for count in range(len(intensity.data)):
            self.data.iloc[count, :] = self.data.iloc[count, :]*(1/intensity.data.values[count])
        print('Done')
        #return intensity
    
    def get_name(self):
        """
        gets the name of the hyper object

        Returns
        -------
        string
            name of the hyper object.

        """
        return self.name
    
    def get_number(self):
        """
        Return the number of spectra the hyper object

        Returns
        -------
        int
            number of spectra of hyperobject.data.

        """
        return len(self.data.index)
    
    def append(self, hyper):
        """
        Append an hyperobject to hyperobject base

        Parameters
        ----------
        hyper : hyperobject
            the appending hyperobject.

        Returns
        -------
        None.

        """
        self.data = pd.concat([self.data, hyper.data])
        self.position = pd.concat([self.position, hyper.position])
        
        list1 = self.label.to_numpy().tolist()
        list2 = hyper.label.to_numpy().tolist()
        self.label = pd.Series(list1 + list2)
        self.label = self.label.rename('label')
        self.reset_index()
        
    def reset_index(self):
        """
        Reset the index of whole hyperobject

        Returns
        -------
        None.

        """
        self.data = self.get_data().reset_index(drop = True)
        self.position = self.get_position().reset_index(drop = True)
        self.label = self.get_label().reset_index(drop = True)
        
    def concat(self, list_hyper):
        """
        concataneta hyperobjects to base hyperobject with same wavenumber axis

        Parameters
        ----------
        list_hyper : list of hyperobjects
            concataneting hyperobjects.

        Returns
        -------
        None.

        """
        #aux = hyper_data('aux')
        for count in range(len(list_hyper)):
            self.append(list_hyper[count])
        self.reset_index()
        print('Done')
      
    def mean(self):
        """
        Return the mean of hyperobject categorically (labeled)

        Returns
        -------
        mean : hyperobject
            hyperobject containing the mean of each label pixel.

        """
        aux = hyper_object('aux')
        std_aux = hyper_object('std_aux')
        std = hyper_object('std')
        mean = hyper_object('mean')
        self.label = self.label.astype(str)
        unique = np.unique(self.label)
        print(unique)
        for count in range(len(unique)):
            aux.set_data(pd.DataFrame(self.data[self.label[:] == str(unique[count])].mean()).T)
            std_aux.set_data(pd.DataFrame(self.data[self.label[:] == str(unique[count])].std()).T)
            aux.set_position(pd.DataFrame(np.zeros((1, 2))))
            aux.position.columns = ['x', 'y']
            aux.set_label(unique[count])
            
            std_aux.set_position(pd.DataFrame(np.zeros((1, 2))))
            std_aux.position.columns = ['x', 'y']
            std_aux.set_label(unique[count])
            
            std.append(std_aux)
            mean.append(aux)
        mean.reset_index()
        std.reset_index()
        return (mean)
    
    def vector(self):
        """
        vector normalization

        Returns
        -------
        None.

        """
        #self.data = self.data.dropna(True)
        result = pd.DataFrame(normalize(self.data.values, norm = 'l2'))
        result.columns = self.data.columns
        result.index = self.data.index
        self.data = result
    
    def show(self, fast):
        """
        plot the average and standard deviation of the frame data
        
        Parameters
        ----------
        fast : False or True
            True = average + standard deviation
            False = individual spectra (max 10)

        Returns
        -------
        NONE

        """
        final = []
        fig_size = plot_conditions()
        fig = plt.figure(num = self.name+'inline', figsize = fig_size, dpi = 300)
        average = self.data.mean()
        std = self.data.std()
        axs = plt.subplot(111)
        axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(300))
        axs.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        length = len(self.data.index)
        if fast == False:
            if length < 10:
                plot = self.data.copy().reset_index(drop = True)  
                for count in range (length):
                    axs.plot(pd.to_numeric(average.index).to_numpy(), plot.iloc[count,:].values, linewidth = 0.7, label = self.label.iloc[count], alpha = 0.7)
            else:
                fast = 10
                print('10 Random Plot')
                rand = np.random.randint(0, len(self.data.index), fast)
                plot = self.data.copy().reset_index(drop = True).iloc[rand, :]   
                for count in range (fast):
                    axs.plot(pd.to_numeric(average.index).to_numpy(), plot.iloc[count,:].values, linewidth = 0.7, alpha = 0.7, label = self.label.iloc[rand[count]])
            leg = axs.legend(frameon = False, loc = 'upper left', bbox_to_anchor=(0.95, 1), handlelength=1)
            # get the individual lines inside legend and set line width
            for line in leg.get_lines():
                line.set_linewidth(4)
        else:
            axs.fill_between(pd.to_numeric(average.index).to_numpy(), average.add(std).values, average.subtract(std).values, alpha=0.30, color = 'k')  
            axs.plot(pd.to_numeric(average.index).to_numpy(), average.values, 'k', linewidth = 0.7)

        axs.set_xlabel('Wavenumber (cm$^{-1}$)')
        axs.set_ylabel('Intensity')  
        plt.tight_layout()
        plt.show()
            
    def select_index(self, lista):
        """
        return the chosen index of the hyper object

        Parameters
        ----------
        lista : list
            name of the indexes.

        Returns
        -------
        Pandas DataFrame
            the matching data.

        """
        return (self.data.iloc[lista, :].copy())
    
    def select_label(self, lista):
        """
        return the chosen label of hyperobject

        Parameters
        ----------
        lista : list  
            name of labels in hyperobject.label.

        Returns
        -------
        aux : hyperobject
            chosen label hyperobject.

        """
        pos = pd.DataFrame(np.ones((len(lista), len(self.position.columns))))

        if len(lista) > 1: 
            selected = pd.DataFrame(np.ones((len(lista), len(self.data.columns))))
            count = 0
            for li in lista:
                selected.iloc[count, :] = pd.DataFrame(self.data[self.label[:] == li])
                count+=1
        else:
            print('single')
            selected = []
            pos = []
            for count1 in range(len(self.label)):
                if self.label.iloc[count1] == lista[0]:
                    selected.append(self.data.iloc[count1].copy())
            selected = pd.DataFrame(selected)
        selected.columns = self.data.columns
        pos.columns = self.position.columns
        aux = hyper_object('selection')
        aux.set_data(selected)
        aux.set_label(lista)
        aux.set_position(pd.DataFrame(np.zeros((len(lista), 2)), columns = ['x', 'y']))
        return aux

    
    def kmeans(self, num_clusters):
        """ Compute Kmeans++ clustering

        Parameters
        ----------
        num_clusters :  int or 'auto'
            number of desired clusters. 
            'auto' finds the number or cluster automatically by silloute method as long as the clusters are distiguisble

        Returns
        -------
        None

        """
        silhouette_coefficients = []
        send = self.data.copy()
        scaled_features = self.data.copy().values
        pre_clusters = 1
        #num_silhoutte =  0
        if num_clusters == 'auto':
            print('auto mode')
            # Notice you start at 2 clusters for silhouette coefficient
            kmeans_kwargs = {
                "init": "k-means++",
                "n_init": 10,
                "max_iter": 300,

            }
        
            try:
                for k in range(2, 8):
                    kmeans = KMeans(n_clusters=k, **kmeans_kwargs )
                    kmeans.fit(scaled_features)
                    score = silhouette_score(scaled_features , kmeans.labels_, metric = 'euclidean')
                    silhouette_coefficients.append(score)
                    #visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
                    #visualizer.fit(X)
                num_silhoutte =  max(silhouette_coefficients)
                
            except ValueError:
                print ('Error : no clusters')
        
            for j in range(len(silhouette_coefficients)):
                if (num_silhoutte == silhouette_coefficients[j]):
                    pre_clusters+=j
            print('coefficients: ', np. around(silhouette_coefficients, 4))
            pre_clusters+=1
            print('Num clusters: ', pre_clusters)

        else:
            pre_clusters = num_clusters
            
        if pre_clusters > 1:
            kmean = KMeans(algorithm='auto', 
                    copy_x=True, 
                    init='k-means++', # selects initial cluster centers
                    max_iter=300,
                    n_clusters = pre_clusters, 
                    n_init=10, 
                    random_state=1, 
                    tol=0.0001, # min. tolerance for distance between clusters
                    verbose=0)
            
            kmean.fit(scaled_features)        
            centers = pd.DataFrame(kmean.cluster_centers_)
            centers.columns = self.data.columns
            
            #Selecting reduced area
            labels = pd.Series(kmean.labels_)
            labels.index = self.data.index
            labels = labels.add(1)
            #labels.rename('label')
            
            self.label = labels
            self.label = self.label.rename('label') 
            #return (labels, centers)
    
    def get_wavenumber(self):
        """
        Return the wavenumber

        Returns
        -------
        array with the wavenumber axis.

        """
        return pd.Series(pd.to_numeric(self.data.columns))
        
    def set_wavenumber(self, series):
        """
        Set the wavenumber calibration to the columns

        Parameters
        ----------
        series : Pandas Series
            wavenumber axis.

        Returns
        -------
        None.

        """
        try:
            self.data.columns = np.round(series.values, 2)
        except:
            print('Failure')
        
    def show_spectra(self, enable, center, colors):
        """
        Show the labeled spectra in the hyper_object by mean of the labeled objects

        Parameters
        ----------
        enable : float
            0: no peaks and larger than 0 adjusts the prominence for finding peaks.
        center : float
            0: no baseline, -1: draw baseline at 0 and > 1 creates and offset among the spectra.
        colors : list of strings or 'auto'
            colors for the labeled data.

        Returns
        -------
        none

        """
        nor = 0
        path = None
        type_file = 'png'
        
        values = self.label.unique()
        if len(values) > 10:
            print('Warning: too many spectra')
        indices = []
        final = []
        fig_size = plot_conditions()
        fig = plt.figure(num = self.name, figsize = fig_size, dpi = 300)
        axs = plt.subplot(111)
        average = pd.DataFrame()
        normalized_avg = pd.DataFrame()
        normalized_std = pd.DataFrame()
        std = pd.DataFrame()
        concat = pd.DataFrame()
        axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(300))
        axs.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100))
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        if colors == 'auto':
            colormap = cm.get_cmap('hsv')
            norm = colors_map.Normalize(vmin=0, vmax=len(values))
            colors = colormap(norm(range(len(values))))
        if len(values) > 1:
            for count in range(len(values)):
                frame = self.data[self.label[:] == values[count]]
                print(values)
                if len(frame.index) > 1:
                    average = frame.mean()
                    std = frame.std()
                    maximum_avg = average.max()
                    minimum_avg = average.min()
                    normalized_avg = average
                    normalized_std = 0
                    axs.plot(pd.to_numeric(self.data.columns), normalized_avg + center*count, label = values[count], linewidth = 0.7, color = colors[count], alpha = 0.7)
                    leg = axs.legend(frameon = False, loc = 'upper left', bbox_to_anchor=(0, 1), handlelength=0.4)
                    for line in leg.get_lines():
                        line.set_linewidth(4)    
                else:    
                    average = frame.mean()
                    maximum_avg = average.max()
                    minimum_avg = average.min()     
                    normalized_avg = average
                    axs.plot(pd.to_numeric(frame.columns), normalized_avg + center*count, label = values[count], linewidth = 0.7, color = colors[count], alpha = 0.7)
                    leg = axs.legend(frameon = False, loc = 'upper left', bbox_to_anchor=(0, 1), handlelength=0.4)
                    for line in leg.get_lines():
                        line.set_linewidth(4)
                
                stick = pd.DataFrame(average).T
                stick['label'] = values[count]
                concat = pd.concat([concat, stick])
                if center == -1:
                    axs.plot(pd.to_numeric(frame.columns), np.zeros(len(frame.columns))+0.5, linewidth = 0.4, color = 'grey')

                if enable > 0:
                    peak_finder(count, axs, normalized_avg + center*count, enable, colors[count], 0)
            axs.set_xlabel('Wavenumber (cm$^{-1}$)')
            axs.set_ylabel('Intensity')
            fig.canvas.set_window_title(self.name) 
            final = concat.reset_index(drop = True)
            unmix = hyper_object('label' + self.name)
            data = final.drop(columns = 'label')
            unmix.set_data(data)
            unmix.set_label(final['label'])  
        else:
            axs.plot(pd.to_numeric(self.data.columns), self.data.iloc[0,:] + center, label = self.name, linewidth = 0.7, color = colors[0], alpha = 0.7)
            if enable > 0:
                peak_finder(0, axs, self.data.iloc[0,:] + center, enable, colors[0], 0)    
            axs.set_xlabel('Wavenumber (cm$^{-1}$)')
            axs.set_ylabel('Intensity')
        plt.tight_layout()

    def select_spectrum(self, label):
        """
        Get a selected spectrum

        Parameters
        ----------
        labels : string
            the name of the label.

        Returns
        -------
        None.

        """
        table = self.data[self.label == label].copy()
        new = create_empty_hyper_object(pd.DataFrame(table).T)
        new.set_label(label)
        return new
        
    def clean_data(self):
        """
        Removes the data out of label index

        Returns
        -------
        None.

        """
        self.data = self.data.iloc[self.label.index, :]
        
    def show_profile(self, colors):
        """
        Plots the abudance results as a profile 

        Parameters
        ----------
        colors : list of strings or 'auto'
            colors for the plotting.

        Returns
        -------
        unmix : TYPE
            DESCRIPTION.

        """
        data = self.data.copy()
        values = self.label.unique()
        final = []
        fig_size = plot_conditions()
        fig = plt.figure(num = self.name+'inline', figsize = fig_size, dpi = 300)
        axs = plt.subplot(111)
        average = pd.DataFrame()
        normalized_avg = pd.DataFrame()
        normalized_std = pd.DataFrame()
        std = pd.DataFrame()
        concat = pd.DataFrame()
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        if colors == 'auto':
            colormap = cm.get_cmap('hsv')
            norm = colors_map.Normalize(vmin=0, vmax=len(values))
            colors = colormap(norm(range(len(values))))
        if len(values) > 1:
            data = data.T
            for count in range(len(values)):
                frame = data[self.label[:] == values[count]]
                print(values)
                if len(frame.index) > 1:  
                    average = frame.mean()
                    normalized_avg = average                                            
                    axs.plot(pd.to_numeric(self.data.columns*self.resolution), normalized_avg, label = values[count], linewidth = 0.7, color = colors[count])
                    axs.legend([values[count]], frameon = False, loc = 'upper left')
                else:    
                    average = frame.mean()
                    normalized_avg = average
                    axs.plot(pd.to_numeric(frame.columns*self.resolution), normalized_avg, label = values[count], linewidth = 0.7, color = colors[count])
                    axs.legend([values[count]], frameon = False, loc = 'upper left')
                stick = pd.DataFrame(average).T
                stick['label'] = values[count]
                concat = pd.concat([concat, stick])
                
        else:
            axs.plot(pd.to_numeric(self.data.index*self.resolution), self.data.values, linewidth = 0.7, color = 'k')
            concat = pd.concat([self.data, self.label], axis = 1)
        plt.legend(frameon = False, loc=2)
        axs.set_xlabel('z [mm]')
        axs.set_ylabel('Intensity')
        fig.canvas.set_window_title(self.name)
        final = concat.reset_index(drop = True)
        unmix = hyper_object('label' + self.name)
        data = final.drop(columns = 'label')
        unmix.set_data(data)
        unmix.set_label(final['label'])
        plt.tight_layout()
        
    def covariance(self, contamination):
        """ DEbbuging Set the contamination value.
        
        The value represents the porcentage of outliers in the dataset for 
        removing by minimum covarience determinat
        
        Parameters:
        -----------------
            temp : float
                the contamination value
        Returns
        ----------------
        no value
        """ 
        #Identifying outliers
        estimator = EllipticEnvelope(contamination = contamination/100)
        prediction = estimator.fit_predict(self.data)
        #clean data
        index = prediction != -1
        new_data = self.data[index, :]
        new_position = self.position[index, :]
        new_cluster = self.label[index, :]
        self.data = new_data
        self.position = new_position
        self.label = new_cluster
            
    def hca(self, distance, linkage, dist, p):
        """ Compute hierchical component analysis and plot dendogram.

        Parameters
        ----------
        distance :  string
            type of pairwise distance calculation.
        linkage : string
            it may be 'ward', 'single' or 'complete'.
        dist : float
            distance for cutting the vertical distances of the dendogram.
        p : int
            number of brances at the end of the dendogram.

        Returns
        -------
        None.

        """
        type_file = 'png'
        model = AgglomerativeClustering(distance_threshold = dist, n_clusters=None, affinity = distance, linkage = linkage)
        model = model.fit(self.data.values)    
        if p == None:
            plot_dendrogram(sch.linkage(self.data.values, method=linkage), color_threshold = dist, max_d = dist, leaf_rotation=90, labels=model.labels_, above_threshold_color='grey')

        else:
            plot_dendrogram(sch.linkage(self.data.values, method=linkage), truncate_mode='level', p=p, color_threshold = dist, max_d = dist, leaf_rotation=90, labels=model.labels_, above_threshold_color='grey')
        
        labels = pd.Series(model.labels_)
        labels = labels.add(1)
        labels.index = self.data.index
        self.label = labels.rename('label')
        print('Num clusters :', model.n_clusters_)
        
    def pls_lda(self, num_components_pls, nor, percentage):
        """
        Performs double supervised pls-lda as long as there are more than 2 classes (or labels).
        
        Parameters
        ----------
        num_components_pls : int
            number of expected components.
        nor : bool
            True aplies normalization otherwise no normalization.
        percentage: float
            0 - 1 (how much percentage for training data)
            1 doesnt split the data and no statistics

        Returns
        -------
        scores: hyperobject
            scores of pls_lda 
        loadings: hyperobject
            loadings of pls_lda

        """
        copy = self.label.copy()
        if self.label.empty == 1:
            exit('Error: No labels')
        if len(self.label.unique()) == 1:
            exit('Error: Only 1 label in the dataset')
        if type(self.label[0]) == str:
            print('Warning: The labels are string')
            unique = copy.unique()
            number = np.arange(len(unique))
            for count in range(len(unique)):
                for count1 in range (len(copy)):
                    if self.label.iloc[count1] == unique[count]:
                        copy.iloc[count1] = number[count]
            copy = pd.Series(copy, dtype = int)
        norm = self.data.copy()
        pls = PLSRegression(n_components = num_components_pls, scale = nor)
        
        #return pls
        x_r, y_r = pls.fit_transform(norm, copy.values)
        
        if percentage != 1:
            X_train, X_test, y_train, y_test = train_test_split(x_r, self.get_label().values, train_size = percentage, random_state=42)
        else:
            y_train = self.label.copy()
            X_train = self.data.copy()
            
        unique = np.unique(y_train)
        clf = LinearDiscriminantAnalysis(solver = 'svd', n_components = None)
        x_l = clf.fit_transform(X_train, y_train)
            
        if len(unique) == 2:
            print('Warning: Final number of components is 1 due to number of unique labels is 2. PLS_LDA is not completely done, instead PLS results are returned')
            scores = hyper_object('scores')
            scores.set_data(np.transpose(x_r))
            scores.set_label(np.arange(1, len(scores.data)+1))
        else:  
            print('Variance per component: ', clf.explained_variance_ratio_)
            scores = hyper_object('scores')
            scores.set_data(np.transpose(x_l))
            scores.set_label(np.arange(1, len(scores.data)+1))
            if percentage != 1:
                y_pred = clf.predict(X_test)
                print(classification_report(np.array(y_test), np.array(y_pred), labels = unique))
    
        loadings = hyper_object('loadings')
        loadings.set_data(np.transpose(pls.x_loadings_))
        loadings.set_wavenumber(pd.to_numeric(self.data.columns))
        loadings.set_label(np.arange(1, num_components_pls+1))
        print('Number of auto final LDA components :', len (scores.label))
        return(scores, loadings)
        
    def save_data(self, file, name):
        """
        Saving hyper_object data.

        Parameters
        ----------
        file : path directory
            place for saving the file.
        name : string
            name of the saving file.

        Returns
        -------
        None.

        """
        name = name
        self.label = self.label.rename('label')
        result = pd.concat([self.data, self.position, self.label], axis = 1)
        gfg_csv_data = pd.DataFrame(result).to_csv(file + '_' + name + '.csv', index = False, header = True) 

    def save_data_xz(self, file, name):
        """
        Saving hyper_object data

        Parameters
        ----------
        file : path directory
            place for saving the file.
        name : string
            name of the saving file.

        Returns
        -------
        None.

        """
        name = name
        self.label = self.label.rename('label')
        result = pd.concat([self.data, self.position, self.label], axis = 1)
        gfg_csv_data = pd.DataFrame(result).to_csv(file + '_' + name + '.csv.xz', index = False, header = True, compression='xz') 
        
    def pca(self, num_components, nor):
        """ Compute pca analysis
        
        Parameters
        ----------
        num_components : int
            number of components.
        
        nor : bool
            application of normalization

        Returns
        -------
        Print the percentage of coovarience
        
        Scores : hyper_object
            array of the components.
        
        Loadings : hyper_object
            array of the loadings

        """
        if self.label.empty == 1:
           print('Error: No labels')
           exit()
        path = None
        color = 'auto'
        unique = self.label.unique()
        length = len(unique)
        transformer = PCA(n_components = num_components)
        if nor == True:
            norm = StandardScaler().fit_transform(self.data)
        else:
            norm = self.data.copy()
        components = transformer.fit(norm)
        X_transformed = transformer.transform(norm)
        aux = 0
        for count in range(len(transformer.explained_variance_ratio_)):
            aux = aux + transformer.explained_variance_ratio_[count]
        print('variance :', aux)
        scores =  hyper_object('scores')
        scores.set_data(np.transpose(X_transformed))
        scores.set_label(np.arange(1, num_components+1))
        scores.set_position(pd.DataFrame(np.zeros((num_components, 2)), columns = ['x', 'y']))
        loadings = hyper_object('loadings')
        loadings.set_data(components.components_)
        loadings.set_wavenumber(pd.to_numeric(self.data.columns))
        loadings.set_label(np.arange(1, num_components+1))
        loadings.set_position(pd.DataFrame(np.zeros((num_components, 2)), columns = ['x', 'y']))
        return (scores, loadings)
        
    def read_1064_3D(self, path_file, path_calibration, resolution, resolutionz):
        """
        still debugging

        Parameters
        ----------
        path_file : TYPE
            DESCRIPTION.
        path_calibration : TYPE
            DESCRIPTION.
        resolution : TYPE
            DESCRIPTION.
        resolutionz : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        calibration_raw = pd.read_table(path_calibration, sep=',', skiprows= 3, usecols = range(514))
        data_raw = pd.read_table(path_file + '/data.txt', sep='\t', lineterminator='\n', usecols = range(515))#, skiprows=[0]);
        #Correction data
        if data_raw.iloc[0,0] == 1 or data_raw.iloc[0,0] == 1:
            data_raw = data_raw.drop(index = 0)
        dark = pd.read_table(path_file + '/dark.txt', sep='\t', lineterminator='\n', header = None, usecols = range(512), skiprows=[0,1]);
        mean = dark.mean()
        data = data_raw.iloc[:, :512]
        data.columns = np.arange(512)
        pos = data_raw.iloc[:, 512:515]
        calibration_data = calibration_raw.iloc[:, :512]  
        pb = pd.read_table( path_file+ '/pb.txt', sep='\t', lineterminator='\n', skiprows=[0], header = None, usecols = range(512));
        pbd = pd.read_table( path_file + '/pbd.txt', sep='\t', lineterminator='\n', skiprows=[0], header = None, usecols = range(512));
        diff = data.subtract(mean)
        diffp = pb.subtract(pbd)
        pre_result = pd.DataFrame(diff.values - diffp.values[0])
        pre_result.columns = calibration_data.columns   
        pos.columns = ['z', 'y', 'x']
        pos.index = pre_result.index
        pos['y'] = pos.iloc[::-1,1].values
        self.position = pos.copy()
        self.data = pre_result.copy()
        self.original = pre_result
        #print(self.original)
        self.resolution = resolution
        self.resolutionz = resolutionz
        max_m = pd.to_numeric(self.position['x'])
        self.m = int(max_m.max() + 1)
        max_n = pd.to_numeric(self.position['y'])
        self.n = int(max_n.max() + 1)
        max_l = pd.to_numeric(self.position['z'])
        self.l = int(max_l.max() + 1)
        self.label = self.data.iloc[:, 0].copy()
        self.label[:] = self.name
        self.data.columns = np.round(self.data.columns, 2)

    def read_mat(self, path):
        """
        Read a raw mat file

        Parameters
        ----------
        path : string
            path and name of the mat file.

        Returns
        -------
        matrix with lists of the elements in mat file.

        """
        return (loadmat(path))

    # def read_mat_holomap(self, path, res):
    #     """
    #     still debugging

    #     Parameters
    #     ----------
    #     path : TYPE
    #         DESCRIPTION.
    #     res : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
    #     #***** Reading mat Hyperspectral data *******
    #     annots = loadmat(path)
    #     aux = annots['react_data']
    #     m = len(aux[0])
    #     n = len(aux[0][0])
    #     p = len(aux)
    #     wave = pd.DataFrame(annots['xaxis'].T)
    #     matrix = np.ones((m*n, p))
    #     position = np.ones((m*n, 2))
    #     #return m, n, p, matrix, aux, position
    #     for count0 in range(m):
    #         for count1 in range(n):
    #                 for count2 in range(p):
    #                     try:
    #                         matrix[count0+count1*m][count2] = aux[count2][count0][count1]
    #                     except:
    #                         print(count2)
    #                 position[count0+count1*m][0] = count1
    #                 position[count0+count1*m][1] = count0
                    
    #     matrix = pd.DataFrame(matrix).reset_index(drop = True)
    #     matrix.columns = wave.iloc[:, 0]
    #     self.data = matrix
    #     self.original = self.data.copy()
    #     self.position = pd.DataFrame(position).rename(columns = {0:'x', 1:'y'})
    #     self.n = n
    #     self.m = m
    #     self.resolution = res
    #     print('loaded')
        
    def set_resolution(self, resolution):
        """
        Set the dedired resolution

        Parameters
        ----------
        resolution : float
            the x and y resolution.

        Returns
        -------
        None.

        """
        self.resolution = resolution
        print (self.name)
        
    def set_resolutionz(self, resolution):
        """
        Set the dedired resolution

        Parameters
        ----------
        resolution : float
            the x and y resolution.

        Returns
        -------
        None.

        """
        self.resolutionz = resolution
        print (self.name)
    
    def show_intensity_3d(self, threshold):
        """
        The method shows the intenisty 3d scatter map

        Parameters
        ----------
        threshold : float
            Removes low intensity concentration scatter points.

        Returns
        -------
        None.

        """
        aux = hyper_object('intensity')
        for count in range(len(self.data)):
            line = self.data.iloc[count, :].copy()
            for count in range(len(line)):
                value = line.iloc[count]
                if value < threshold:
                    line.iloc[count] = np.nan
            aux.set_data(self.data.T)
            aux.label = line
            aux.set_position_3d(self.position)
            #return aux
            aux.scatter_3D('intensity')
            
    def show_intensity(self, cmap):
        """
        it shows the intensity 1d hyper_object

        Parameters
            ----------
            cmap : special string 
                color map for the plotting. Use: 'inferno', 'viridis', 'plasma', 'gray'

        Returns
        -------
        None.

        """
        fig_size = plot_conditions()
        interpolation = None
        size_x = self.resolution*self.m
        size_y = self.resolution*self.n
        square = self.data.copy()
        if len(square.columns) > 1:
            for count0 in range(len(self.data.columns)):
                selection = square.iloc[:, count0]
                fig = plt.figure(num = 'map: ' + self.name + '_' + str(self.label.iloc[count0]), figsize = fig_size, dpi = 300)    
                aux = np.zeros((self.m, self.n))
                for count1 in range(len(selection.index)):
                    try:
                        xi = self.position.iloc[selection.index[count1], 0]
                        yi = self.position.iloc[selection.index[count1], 1]
                    except:
                        print('removed')
                    try:
                        aux[int(xi)][int(yi)] = selection.iloc[count1] 
                    except:
                        print(xi, yi, len(aux))
                plt.imshow(np.rot90(aux, 1, axes = (0, 1)), extent = [0, size_x, 0, size_y], cmap = 'inferno', interpolation = interpolation)
                plt.xlabel(' Size [mm]')
                plt.ylabel(' Size [mm]')
                cbar = plt.colorbar()
                cbar.ax.get_yaxis().labelpad = 15
                cbar.ax.set_ylabel('Intensity', rotation = 90)
                plt.tight_layout()
        else:
            fig = plt.figure(num = 'map: ' + self.name + str(0), figsize = fig_size, dpi = 300)    
            aux = np.zeros((self.m, self.n))
            for count1 in range(len(square.index)):
                try:
                    xi = self.position.iloc[square.index[count1], 0]
                    yi = self.position.iloc[square.index[count1], 1]
                except:
                    print('removed')
                try:
                    aux[int(xi)][int(yi)] = square.iloc[count1] 
                except:
                    print(xi, yi, len(aux))
            plt.imshow(np.rot90(aux, 1, axes = (0, 1)), extent = [0, size_x, 0, size_y], cmap = 'inferno', interpolation = interpolation)
            plt.xlabel(' Size [mm]')
            plt.ylabel(' Size [mm]')
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel('Intensity', rotation = 90)
            plt.tight_layout()
        #return aux
    
    def abundance(self, mean, constrain):
        """
        It calculates the abudance of the mean spectra onto the spectral map

        Parameters
        ----------
        mean : hyperobject
            hyperobject with the endmembers for abundance calculation.
            
        constrain: string
            'NNLS' or 'OLS'
        Returns
        -------
        abu : hyperobject
            concentration of each component.

        """   
        path = None
        M = self.data.copy().to_numpy()
        U = mean.data.values
        if constrain == 'NNLS':
            aux = NNLS(M, U)
        else:
            aux = OLS(M, U)   
        abundance = pd.DataFrame(aux)          
        abu = hyper_object(self.name + '_abundance')
        abu.set_data(abundance)
        abu.set_label(mean.get_label())        
        abu.reset_index()
        if len(self.position.columns) == 2:
            abu.set_position(self.position)
        else:
            abu.set_position_3d(self.position)
        return abu
      
    def clean_label(self):
        """
        cleans the unmatched label and data indexes

        Returns
        -------
        None.

        """
        self.label = self.label.iloc[self.data.index]
        
    def get_resolution(self):
        """
        It returns the x and y resolution

        Returns
        -------
        float
            spatial resolution of the hyperobject.

        """
        return self.resolution
        
    def vca(self, num):
        """
        It calculates the vertex componenet analysis

        Parameters
        ----------
        num : int
            number of expected components.

        Returns
        -------
        unmix : hyperobject
            endmembers.

        """
        endmember, index, concentration = vca(self.data.copy().T.values, num)
        concat = self.data.iloc[index, :].sort_index()
        #return concat
        #concat.T.plot()
        unmix = hyper_object('vca')
        unmix.set_data(concat)
        aux = pd.Series(concat.index)
        aux.index = concat.index
        #print(aux.index)
        unmix.set_label(pd.Series(aux.index))
        matrix = pd.DataFrame(np.zeros((len(aux.index),2)))
        matrix.index = concat.index
        matrix.rename = ['x', 'y']
        unmix.set_position(matrix)
        return (unmix)
        
    def show_map(self, colors, interpolation, unit_in):
        """
        It plots the label data in the hyperobject

        Parameters
        ----------
        colors : string list of colors or 'auto'
            the colors for the map visualization.
        interpolation : string
            kind of pixel interpolation: 'None', 'nearest', 'bicubic', 'gaussian'.
        unit_in : int
            the plotting scale, 1 = mm and 1000 um.

        Returns
        -------
        colors : TYPE
            DESCRIPTION.

        """
        path = None
        type_file = 'png'
        
        if unit_in == 1:
            unit = 'mm'
            
        if unit_in == 1000:
            unit = '\u03BCm'        ##############################################################
        values = []
        fig_size = plot_conditions()
        #print (self.n)
        fig = plt.figure(num = 'map: ' + self.name, figsize = fig_size, dpi = 300)    

        n_bin = len(colors)  # Discretizes the interpolation into bins
        cmap_name = 'my_list'
        size_x = self.resolution*self.m*unit_in
        size_y = self.resolution*self.n*unit_in
        aux = np.zeros((self.m, self.n))
        cluster = self.label.copy()
        unique = self.label.unique()
        aux[:] = np.nan
        rise = 0
        if colors == 'intensity' or len(unique) > 20:
            cmap = 'inferno'
            rise = 1
        else:
            if colors == 'auto':
                cmap = cm.get_cmap('tab20', len(unique))
                #norm = colors_map.Normalize(vmin=0, vmax=len(values))
                #map = colormap(norm(range(len(values))))

            else:                
                cmap = LinearSegmentedColormap.from_list(cmap_name, colors[:len(unique)], N = n_bin*1)
        
        if(type(self.label.iloc[0]) == str):
            for count in range (0, len(unique)):
                cluster[cluster.iloc[:] == unique[count]] = count + 1

        for count1 in range(len(self.data.index)):
            try:
                xi = self.position.iloc[self.data.index[count1], 0]
                yi = self.position.iloc[self.data.index[count1], 1]
            except:
                print('removed')
            try:
                aux[int(xi)][int(yi)] = cluster.iloc[count1] 
            except:
                print(xi, yi, len(aux))
                
        boundaries = cluster.unique()
        im = plt.imshow(np.rot90(aux, 1, axes = (0, 1)), extent = [0, size_x, 0, size_y], cmap = cmap, interpolation = interpolation)
        plt.xlabel(' Size ['+unit+']')
        plt.ylabel(' Size ['+unit+']')

        if rise == 0:
            values = cluster.unique()
            label = self.label.unique()
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color = colors[i], label="{l}".format(l= label[i]) ) for i in range(0, len(values)) ]
            plt.legend(handles = patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0, frameon = False)
        
        else:
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel('Intensity', rotation=90)
        plt.tight_layout()
        return colors
    
    def cluster_cluster(self, point, num):
        """
        It applies kmeans clustering to the chosen label

        Parameters
        ----------
        point : string or int
            the label name.
        num : int
            number of clusters.

        Returns
        -------
        None

        """
        # A list holds the silhouette coefficients for each k
        silhouette_coefficients = []
        #self.data = self.data.dropna()
        send = self.data[self.label[:] == point].copy()
        #scaler = StandardScaler()
        scaled_features = self.data[self.label[:] == point].copy()
        pre_clusters = num
        if pre_clusters > 1:
            kmean = KMeans(algorithm='auto', 
                    copy_x=True, 
                    init='k-means++', # selects initial cluster centers
                    max_iter=300,
                    n_clusters = pre_clusters, 
                    n_init=10, 
                    random_state=1, 
                    tol=0.0001, # min. tolerance for distance between clusters
                    verbose=0)
            
            kmean.fit(scaled_features)        
            centers = pd.DataFrame(kmean.cluster_centers_)
            centers.columns = self.data.columns
            
            #Selecting reduced area
            labels = pd.Series(kmean.labels_)
            labels.index = send.index
            labels = labels.add(point*10)
            labels = labels.rename('label')
        
            self.label[send.index] = labels
            
            #return self.label
        else:
            print('Error: number of clusters found is 1!!')
        
    # def combine_label(self, before, after):
    #     """
    #     It combines the label names

    #     Parameters
    #     ----------
    #     before : string list
    #         the current label names.
    #     after : string list
    #         the new label name for the current label names.

    #     Returns
    #     -------
    #     None.

    #     """
    #     cluster = self.label.copy()
        
    #     for count in range(len(before)):
    #         cluster[cluster.iloc[:] == before[count]] = after[count]
                
    #     cluster = cluster.rename('label')
    #     concat = pd.concat([self.data, cluster, self.position], axis = 1).dropna()
        
    #     self.data = concat.iloc[:, :len(self.data.columns)]
    #     self.label = concat['label']
    #     self.position = concat[['x', 'y']]
    
    def remove_label(self, before):
        """
        Removal of spectra labeled

        Parameters
        ----------
        before : list of label names
            the labels for removal.

        Returns
        -------
        None.

        """
        cluster = self.label.copy()
        after = before.copy()
        try:
            for count in range(len(before)):
                cluster[cluster.iloc[:] == before[count]] = np.nan
            
            cluster = cluster.rename('label')
            concat = pd.concat([self.data, cluster, self.position], axis = 1).dropna()
            
            self.data = concat.iloc[:, :len(self.data.columns)]
            self.label = concat['label']
            self.position = concat[['x', 'y']]
        except:
            print('No Correct Input')
        self.reset_index()
        #self.clean_label()
        
    def remove_spectrum(self, index):
        """
        It removes the spectrum specified by index of the hyperobject

        Parameters
        ----------
        index : int
            index of the spectrum.

        Returns
        -------
        None.

        """
        self.data = self.data.drop(index = index)
        self.position = self.position.drop(index = index)
        self.label = self.label.drop(index = index)
        
    def spikes(self, limit, size):
        """
        Removal of spikes

        Parameters
        ----------
        limit : float
            threshold (standard is 7).
        size : int
            size of the window (it depends on how wide the spike is).

        Returns
        -------
        None

        """
        result = np.zeros(len(self.data.index)*len(self.data.columns)).reshape(len(self.data.index), len(self.data.columns))
        copy = self.data.copy()
        length = len(copy.index)
        for count in range (length):
            array = np.array(copy.iloc[count, :])
            spike = fixer(array, size, limit)
            #print('Spectrum: ', count, '/', length, 'type' , np.unique(np.isnan(spike)))
            #print(spike)
            negatives = np.isnan(spike)
            for count1 in range(size, len(negatives)-size):
                if negatives[count1] == 1:
                    w = np.arange(count1-size, count1+size+1) # we select 2 m + 1 points around our spike
                    w2 = w[negatives[w] == 0] # From such interval, we choose the ones which are not spikes
                    if len(w2)>1:
                        spike[count1] = np.mean(spike[w2])
                    else:
                        spike[count1] = spike[count1-1]
            result[count] = spike.T
            print('Spectrum: ', count+1, '/', length)

        frame = pd.DataFrame(result)
        frame.columns = self.data.columns
        frame.index = self.data.index

        self.data = frame.dropna()
      
    def label_spikes(self, cluster, limit, size):
        """
        It applies removal of spikes to the specified label

        Parameters
        ----------
        cluster : string list
            label names.
        limit : float
            threshold (standard 7).
        size : int
            window size (the larger the wider the spike).

        Returns
        -------
        None.

        """
        selection = 0
        copy = self.data.copy()
        for count0 in range (len(cluster)):
            for count in range (len(copy.index)):
                selection = self.label.iloc[count]
                #print(selection)
                if cluster[count0] == selection:
                    array = np.array(copy.iloc[count, :])
                    spike = fixer(array, size, limit)
                    result = pd.Series(spike.T)
                    self.data.iloc[count, :] = result.values
        self.data = self.data.dropna()

    # def mcr_als(self, spectra, constrain, iterations):
    #     """
    #     It applies the mcr_als analysis

    #     Parameters
    #     ----------
    #     spectra : hyperobject 
    #         guess spectra.
    #     iterations : int
    #         number of iterations for getting small error.

    #     Returns
    #     -------
    #     resolved : hyperobject
    #         mcr_als endmembers.

    #     """
    #     # If you have an initial estimate of the spectra
    #     D_known = self.data.to_numpy()
    #     wavenumber = self.data.columns
    #     St_known = spectra.get_data().to_numpy()
        
    #     if constrain == 'positive':
    #         mcrar = McrAR(max_iter = iterations)
    #     else:
    #         mcrar = McrAR(max_iter = iterations, tol_increase=1, c_regr='NNLS', st_regr='NNLS', c_constraints=[ConstraintNorm()])

    #     mcrar.fit(D_known, ST=St_known, verbose=True)
        
    #     print('\nFinal MSE: {:.7e}'.format(mcrar.err[-1]))
        
    #     temp = mcrar.ST_opt_
    #     conc = mcrar.C_opt_.T
        
    #     resolved = hyper_object('mcr')
    #     resolved.set_data(temp)
    #     resolved.set_label(resolved.data.index)
    #     resolved.set_wavenumber(pd.to_numeric(wavenumber))
        
    #     abu = hyper_object(self.name + '_abundance')
    #     abu.set_data(conc)
    #     abu.set_label(spectra.get_label())
        
    #     return (resolved, abu)
    
    def show_scatter_3d(self, colors, label, size):
        """
        The method shows a 3d scatter plot of the three first components

        Parameters
        ----------
        colors : string: 'auto'
            colors for the plotting.
        label : Panda Series
            Label of the hyperobject for visualization.
        size : float
            size of the scatter points in the plot.

        Returns
        -------
        None.

        """
        unique = label.unique()
        print('3D scatter')
        fig_size = plot_conditions()
        c = label
        length = len(unique)
        if colors == 'auto':
            colormap = cm.get_cmap('hsv')
            norm = colors_map.Normalize(vmin=0, vmax=length)
            colors = colormap(norm(range(length)))
        
        newcolors = np.ones(len(label), dtype = object)
        newcolors[:] = 'white'
        for count in range(len(label)):
            for count1 in range (len(unique)):
                if label.iloc[count] == unique[count1]:
                    newcolors[label.index[count]] = colors[count1]
        c = newcolors
            
        fig = plt.figure(figsize = fig_size, dpi = 300)        
        ax = fig.add_subplot(projection='3d')

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        x = self.data.iloc[0, :].values
        y = self.data.iloc[1, :].values
        z = self.data.iloc[2, :].values
        
        ax.scatter(x, y, z, c = c, s = size, marker = 'o', alpha = 0.7, edgecolor = 'k', linewidths = 0.1)
        
        patch = []
        for count in range(len(unique)):
            patch.append(plt.Line2D([],[], marker="o", ms=size/2, ls="", mec=None, color=colors[count], label=unique[count]))
       
        ax.legend(handles = patch, loc = 2, borderaxespad=0, frameon = False, facecolor="plum", numpoints=1)
        plt.tight_layout()
        #plt.show()
        
    def splitting(self, label, percentage):
        """
        It splits the data into a training and testing datasets

        Parameters
        ----------
        label : Panda Series
            expected classification labeling
        percentage : float
            percentage % for splitting the data into training

        Returns
        -------
        X_train_h : hyper_object
            training data
        X_test_h : hyper_object
            testing data

        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, label, test_size= 1 - percentage/100, random_state=42)
        
        X_train_h =  hyper_object('X_train')
        X_train_h.set_data(X_train)
        X_train_h.set_label(y_train)
        X_train_h.set_position(pd.DataFrame(np.zeros((len(y_train), 2)), columns = ['x', 'y']))
        
        X_test_h = hyper_object('X_test')
        X_test_h.set_data(X_test)
        X_test_h.set_label(y_test)
        X_test_h.set_position(pd.DataFrame(np.zeros((len(y_test), 2)), columns = ['x', 'y']))
        
        return (X_train_h, X_test_h)
    