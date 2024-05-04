#Others
import sys
import numpy as np
import spc_spectra as spc
import matplotlib as mpl
import matplotlib.pyplot as plt
#import hdbscan

import scipy as sp
import sklearn as sk
import numpy as np
import colorcet as cc
from sklearn.cluster import KMeans

#############################################
# Internal functions
#############################################

def rubber_baseline(data, wavenumber):
    try:
        y = data.copy()
        x = wavenumber
        baseline = rubberband(x, y)
        return baseline
    except:
        print('Error in rubber_baseline')

def plot_conditions():
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
    #plt.rcParams.update({'figure.autolayout': True})

    return fig_size
        
def find_pixel(wavenumber_range, target):
    """
    It finds the position of the pixel in the wavenumber range
    Parameters
    ----------
    wavenumber_range : float vector 1d
        wavenumber axis
    target : float
        peak targeted to find the pixel position
    Returns
    -------
    position : int
        the pixel position of the peak.
    """
    # Find the position of the value closest to the target value
    position = np.argmin(np.abs(wavenumber_range - target))
    return position

def NNLS(M, U):
    """
    NNLS performs non-negative constrained least squares of each pixel
    in M using the endmember signatures of U.  Non-negative constrained least
    squares with the abundance nonnegative constraint (ANC).
    Utilizes the method of Bro.
    Parameters
    ----------
    M  numpy array
        2D data matrix (N x p).
    U numpy array
        2D matrix of endmembers (q x p).
    Returns
    -------
    X  array
        An abundance maps (N x q).
    References
        Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401.
    """
    N, p1 = M.shape
    q, p2 = U.shape

    X = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(U, U.T)
    for n1 in range(N):
        X[n1] = sp.optimize.nnls(MtM, np.dot(U, M[n1]))[0]
    return X

def OLS(M, U):
    """
    NNLS performs linear constrained least squares of each pixel
    in M using the endmember signatures of U.  Non-negative constrained least
    squares with the abundance nonnegative constraint (ANC).
    Utilizes the method of Bro.
    Parameters
        M `numpy array`
            2D data matrix (N x p).
        U `numpy array`
            2D matrix of endmembers (q x p).
    Returns `numpy array`
        An abundance maps (N x q).
    References
        Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401.
    """
    N, p1 = M.shape
    q, p2 = U.shape

    X = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(U, U.T)
    for n1 in range(N):
        # opt.nnls() return a tuple, the first element is the result
        X[n1] = sp.optimization.lsq_linear(MtM, np.dot(U, M[n1]))[0]
    return X

def savitzky_golay(y, window_size=7, order=2):
    return sp.signal.savgol_filter(y, window_size, order)

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
    import random
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
      Ud  = sp.linalg.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
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
           
          Ud  = sp.linalg.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:d]  # computes the p-projection matrix 
          x_p =  sp.dot(Ud.T,Y_o)                 # project thezeros mean data onto p-subspace
                  
        Yp =  sp.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L             
        x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = sp.amax(sp.sum(x**2,axis=0))**0.5
        y = sp.vstack(( x, c*sp.ones((1,N)) ))
    else:
      if verbose:
        print("... Select the projective proj.")
      d = R
      Ud  = sp.linalg.svd(sp.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
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
      w = np.random.rand(R,1);   
      f = w - sp.dot(A,sp.dot(sp.linalg.pinv(A),w))
      f = f / sp.linalg.norm(f)      
      v = sp.dot(f.T,y)
      indice[i] = sp.argmax(sp.absolute(v))
      A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))
    Ae = Yp[:,indice]
    return (Ae,indice,Yp)
  
def snip(raman_spectra,niter):
    #snip algorithm
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

def rubberband(x, y):
    ###rubber band algorithm
    result = (x, y)
    #re = np.array(result)
    v = sp.spatial.ConvexHull(np.array(result).T).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax()]
    return np.interp(x, x[v], y[v])

def create_table(numx, numy):
    """"
    It creates a table
    Parameters
    ----------
    numx  int
        number of cols.
    numy  int
        number of rows.
    Returns
    -------
    table  dataframe
        DESCRIPTION.
    """
    table = np.zeros((numx*numy, 2))
    count1 = 0
    count2 = 0
    for count in range (numx*numy):
        if count1 == numx:
            count1 = 0
            count2+=1
        table[count,0] = count1
        table[count,1] = count2
        count1+=1
    return table

def create_snake_table(numx, numy):
    """
    It creates a snake table

    Parameters
    ----------
    numx  int
        number of cols.
    numy  int
        number of rows.

    Returns
    -------
    table  TYPE
        DESCRIPTION.
    """
    table = np.zeros((numx*numy, 2))
    count1 = 0
    flag = 0
    for count in range (numx*numy):
        if count1 == 0:
            flag = 0
        if count1 == numx:
            flag = 1
        if flag == 0:
            table[count,0] = count1
            count1+=1
        if flag == 1:
            count1-=1
            table[count,0] = count1
    count1 = 0
    count2 = 0
    for count in range (numx*numy):
        if count1 == numx:
            count1 = 0
            count2+=1
        table[count,1] = count2
        count1+=1
    return table

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

def peak_finder(data, wavenumber, prominence=1, distance=10):
    # Normalize the data
    data_norm = data / data.max()

    # Find the peaks
    peaks, _ = sp.signal.find_peaks(data, prominence=prominence, distance=distance)

    # Find the closest indices in the wavenumber array for each peak
    indices = np.abs(np.subtract.outer(wavenumber, peaks)).argmin(0)

    return peaks, indices
# '    keep = []    
#     for item in peaks:
        
#         if digits == 0:
#             axs.annotate(int(wavenumber[item]), xy = (wavenumber[item]+5, height), rotation = 90, size = 8, color = color)
#             axs.axvline(x = int(wavenumber[item]), linestyle='--', linewidth = 0.6, alpha = 0.5, color = color)
#             keep.append(int(wavenumber[item]))
#         else:
#             axs.annotate(np.round(wavenumber[item],digits), xy = (wavenumber[item]+5, height), rotation = 90, size = 8, color = color)
#             axs.axvline(x = wavenumber[item], linestyle='--', linewidth = 0.6, alpha = 0.5, color = color)
#             keep.append(np.round(wavenumber[item] ,digits))
#     return keep'

def modified_z_score(delta_int):
    #spikes removing algorithm
    median_int = np.median(delta_int)
    mad_int = np.median([np.abs(delta_int-median_int)])
    modified_z_scores = 0.6745*(delta_int-median_int)/mad_int        
    return modified_z_scores

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
    E=sp.sparse.eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=sp.sparse.diags(w,0,shape=(m,m))
    A=sp.sparse.csc_matrix(W+(lambda_*D.T*D))
    B=sp.sparse.csc_matrix(W*X.T)
    background=sp.sparse.linalg.spsolve(A,B)
    return np.array(background)

def airPLS(x, strength, porder = 1, itermax = 50):
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
        z=WhittakerSmooth(x,w,strength, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print( 'WARNING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z
     
########################################
##### definition of hyper_object #######
########################################

class intensity_object:
    def __init__(self, name, data, position, resolutionx, resolutiony):
        self.name = name
        self.position = position
        self.data = data.copy()
        self.num_stepsx = int(np.max(position[:,0])) + 1
        self.num_stepsy = int(np.max(position[:,1])) + 1
        self.resolutionx = resolutionx
        self.resolutiony = resolutiony


    ### show the intensity map using position and data
    def show_map(self, xlabel, ylabel, ybarlabel, title, rotate=False, order = 'F'):

        fig, ax = plt.subplots()
        cmap = cc.cm.linear_protanopic_deuteranopic_kbw_5_98_c40

        if rotate:
            plane = np.rot90(self.data.reshape((self.num_stepsx, self.num_stepsy), order=order))
            diameterx = self.num_stepsy * self.resolutiony
            diametery = self.num_stepsx * self.resolutionx
            xticks = np.linspace(0, diameterx, 6)  # Set custom xtick values
            yticks = np.linspace(0, diametery, 6)  # Set custom ytick values
            ax.set(xlabel=xlabel, ylabel=ylabel, xticks=yticks, yticks=xticks)
        else:
            plane = self.data.reshape((self.num_stepsx, self.num_stepsy), order=order)
            diameterx = self.num_stepsx * self.resolutionx
            diametery = self.num_stepsy * self.resolutiony
            xticks = np.linspace(0, diameterx, 6)  # Set custom xtick values
            yticks = np.linspace(0, diametery, 6)  # Set custom ytick values
            ax.set(xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks)

        img = ax.imshow(plane, cmap=cmap)
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(ybarlabel)
        ax.set_title(title)
        plt.show()

    def get_data(self):
        return (self.data)
    
    def get_steps(self):
        return self.num_stepsx, self.num_stepsy
    
    def set_steps(self, stepsx, stepsy):
        self.num_stepsx = stepsx, self.num_stepsy = stepsy

    def get_position(self):
        return (self.position)

    def set_data(self, data):
        self.data = data.copy()

    def set_resolution(self, resolutionx, resolutiony):
        self.resolutionx = resolutionx
        self.resolutiony = resolutiony

    def set_position(self, position):
        self.position = position

class single_object:

    def __init__(self, name, data, wavenumber):
        self.data = np.array(data).copy()  # intensity
        self.name = name # name of the object
        self.wavenumber = np.array(wavenumber).copy() # wavenumber

    def get_wavenumber(self):
        return self.wavenumber.copy()
    
    def copy(self):
        return single_object(self.name, self.data, self.wavenumber)
    
    def set_data (self, data):
        self.data = data.copy()
        
    def spikes_filter(self, limit=7, size=5):
        """
        Removal of spikes
        Parameters
        ----------
        limit  float
            threshold (standard is 7).
        size  int
            size of the window (it depends on how wide the spike is).
        Returns
        -------
        None
        """
        result = self.data.copy()
        copy = self.data.copy()
        spike = fixer(copy, size, limit)
            #print('Spectrum ', count, '', length, 'type' , np.unique(np.isnan(spike)))
            #print(spike)
        negatives = np.isnan(spike)
        for count1 in range(size, len(negatives)-size):
            if negatives[count1] == 1:
                w = np.arange(count1-size, count1+size+1) # we select 2 m + 1 points around our spike
                w2 = w[negatives[w] == 0] # From such interval, we choose the ones which are not spikes
                if len(w2)>1:
                    spike[count1] = np.median(spike[w2])
                    print(spike)
                else:
                    spike[count1] = spike[count1-1]
        result = spike.T
            #print('Spectrum ', count+1, '', length)
        self.data = result
        print('Done')

    def show(self, tittle= 'single', method=True):
        plt.figure()
        plt.plot(self.wavenumber, self.data)
        plt.xlabel('Wavenumber (cm-1)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(self.name)

    def get_intensity(self, wavenumber):
        """
        it gets the intensity at certain wavenumber position

        Parameters
        ----------
        wave  float
            wavenumber.

        Returns
        -------
        intensity Hyper_object
            the data dataframe contains the intensity at the peak.

        """
        index = find_pixel(self.wavenumber, wavenumber)
        return self.data[index]
    
    def get_data(self):
        return (self.data.copy())
    
    def keep_region(self, lower, upper):
        """
        It keeps a region between lower and upper wavenumber
        Parameters
        ----------
        lower  float
            lower wavenumber.
        upper  float
            upper wavenumber.
        Returns
        -------
        None.

        """
        index_lower = find_pixel(self.wavenumber, lower)
        index_upper = find_pixel(self.wavenumber, upper)
        #print(index_lower, index_upper)
        self.data = self.data[index_lower: index_upper]
        self.wavenumber = self.wavenumber[index_lower: index_upper] 

    def airpls_baseline(self, lamda):
        """
        calculate advanced baseline correction airpls
        Parameters
        ----------
        value  float 
            value represents how strong is the fitting.
        Returns
        -------
        None.

        """
        #Baseline correction
        matrix = airPLS(self.data.copy(), lamda)
        correction = self.data - matrix
        self.data = correction
        print('Done')

    def gol(self, window=5, polynomial=3):

        """
         Set the value window, polynomial, order
        
        The savitky-golay filter is applied to the full data set

        Parameters
        ----------
        window  int
            size of neighbor pixels to consider.
        polynomial  int
            order for interpolation.
        order  int
            order of derivation.

        Returns
        -------
        None.

        """
        pre_result = savitzky_golay(self.data, window, polynomial)
        self.data = pre_result
        #print(Done)
    
    def get_area(self, lower, upper):
        """
        Calculate the area under the curve
        
        Parameters
        ----------
        lower  float
            lower wavenumber axis.
        upper  float
            upper wavenumber axis.
        Returns
        -------
        abu  intensity hyper_object
            area under the curve.
        """
        lower_wave = find_pixel(self.wavenumber, lower)
        upper_wave = find_pixel(self.wavenumber, upper)
        ranging = self.data[lower_wave, upper_wave].copy()
        area = np.trapz(ranging)
        return area
    
    def rubber_baseline(self):
        """
        calculate advanced baseline correction rubberband
        Parameters
        ----------
        None.
        Returns
        -------
        None.

        """
        self.data = self.data - rubber_baseline(self.data, self.wavenumber)
    
    def airPLS(self, strength):
        """
        calculate advanced baseline correction airpls

        Parameters
        ----------
        value  float 
            value represents how strong is the fitting.

        Returns
        -------
        None.

        """
        #Baseline correction
        matrix = airPLS(self.data.copy(), strength)
        correction = self.data - matrix
        self.data = correction
        print('Done')
    
class image_object:

    def __init__(self, name, data, wavenumber, position, resolutionx, resolutiony, label):
        self.name = name
        self.position = position
        self.data = data.copy()
        self.num_stepsx = int(np.max(position[:,0])) + 1
        self.num_stepsy = int(np.max(position[:,1])) + 1
        self.resolutionx = resolutionx
        self.resolutiony = resolutiony
        self.label = label
        self.wavenumber = wavenumber

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label.copy()
    
    def find_pixel_position(self, x, y):
        """
        It finds the position of the pixel in the map
        Parameters
        ----------
        x  int
            x position.
        y  int
            y position.
        Returns
        -------
        None.
        """
        index = x + y*self.num_stepsx
        return index

    def crop_image(self, x1, x2, y1, y2):
        index1 = self.find_pixel_position(x1, y1)
        index2 = self.find_pixel_position(x2, y2)
        self.data = self.data[index1:index2, :]
        self.position = self.position[index1:index2, :]
        self.label = self.label[index1:index2]
        self.num_stepsx = x2 - x1
        self.num_stepsy = y2 - y1
        print('Done cropping')

    def get_pixel(self, x, y):
        """
        It selects a pixel from the map
        Parameters
        ----------
        x  int
            x position.
        y  int
            y position.
        Returns
        -------
        None.

        """
        data_tmp = self.data[self.find_pixel_position(x, y),:]
        return single_object('pixel_'+str(x)+'_'+str(y), data_tmp, self.wavenumber)
        #self.label = self.label[index]

    def copy(self):
        return image_object(self.name, self.data, self.wavenumber, self.position, self.resolutionx, self.resolutiony, self.label)
    
    def keep_region(self, lower, upper):
        index_lower = find_pixel(self.wavenumber, lower)
        index_upper = find_pixel(self.wavenumber, upper)
        self.data = self.data[:,index_lower: index_upper]
        self.wavenumber = self.wavenumber[index_lower: index_upper] 

    def gol(self, window=5, polynomial=3):
        """ 
        Set the value window, polynomial, order
        
        The savitky-golay filter is applied to the full data set

        Parameters
        ----------
        window  int
            size of neighbor pixels to consider.
        polynomial  int
            order for interpolation.
        order  int
            order of derivation.
        Returns
        -------
        None.

        """
        pre_result = sp.signal.savgol_filter(self.data, window, polynomial, 0, 1)
        self.data = pre_result

    def get_area(self, lower, upper):
        """
        Calculate the area under the curve
        
        Parameters
        ----------
        lower  float
            lower wavenumber axis.
        upper  float
            upper wavenumber axis.
        Returns
        -------
        abu  intensity hyper_object
            area under the curve.
        """
        lower_wave = find_pixel(self.wavenumber, lower)
        upper_wave = find_pixel(self.wavenumber, upper)
        ranging = self.data[:, lower_wave: upper_wave].copy()
        area = np.trapz(ranging)
        return area
    
    def airpls_baseline(self, lamda):
        """
          
        calculate advanced baseline correction airpls
        Parameters
        ----------
        value  float 
            value represents how strong is the fitting.
        Returns
        -------
        None.
        """
        #Baseline correction
        length = len(self.data)
        matrix = np.zeros((length, len(self.wavenumber)))
        for item in range(length):
            matrix[item] = airPLS(self.data[item,:].copy(), lamda)
            print('progress  ', item + 1, '', length)
                
        correction = self.data - matrix
        self.data = correction
        print('Done')

    def vector_normalization(self):
        """
        Normalization of the data
        Returns
        -------
        None.

        """
        self.data = self.data/np.linalg.norm(self.data, axis=1)[:, None]

    def show_map(self, xlabel='x', ylabel='y', ybarlabel='Intensity', title='Map', order = 'F'):
        
        label_map = self.label.reshape((self.num_stepsx, self.num_stepsy), order=order)
        plt.figure()
        plt.imshow(label_map)

        plt.figure('clusters')

        # Iterate over the cluster labels
        for i in range(len(np.unique(self.label))):
            aux = np.mean(self.data[self.label == i], axis=0)
            plt.plot(self.wavenumber, aux, label=f'cluster {i}')

        plt.legend()

        plt.show()

    def get_data(self):
        return (self.data)
    
    def get_steps(self):
        return self.num_stepsx, self.num_stepsy
    
    def kmeans_cluster(self, num):
        """
        Kmeans clustering
        Parameters
        ----------
        num  int
            number of clusters.
        Returns
        -------
        None.

        """
        kmeans = sk.cluster.KMeans(n_clusters=num, random_state=0).fit(self.data)
        self.label = kmeans.labels_

class multi_object:
    def __init__(self, data, wavenumber, name):
        self.data = data.copy()
        self.wavenumber = wavenumber.copy()
        self.name = name
    
    def keep_region(self, lower, upper):
        index_lower = find_pixel(self.wavenumber, lower)
        index_upper = find_pixel(self.wavenumber, upper)
        self.data = self.data[:,index_lower, index_upper]
        self.wavenumber = self.wavenumber[index_lower, index_upper] 

    def gol(self, window=5, polynomial=3):
        """
         Set the value window, polynomial, order
        
        The savitky-golay filter is applied to the full data set

        Parameters
        ----------
        window  int
            size of neighbor pixels to consider.
        polynomial  int
            order for interpolation.
        order  int
            order of derivation.
        Returns
        -------
        None.

        """
        pre_result = sp.signal.savgol_filter(self.data, window, polynomial, 0, 1)
        self.data = pre_result
        #print(Done)
    def airpls_baseline(self, landa):
        """
        calculate advanced baseline correction airpls
        Parameters
        ----------
        value  float 
            value represents how strong is the fitting.
        Returns
        -------
        None.

        """
        #Baseline correction
        length = len(self.data)
        matrix = np.zeros((length, len(self.wavenumber)))
        for item in range(length):
            matrix[item] = airPLS(self.data[item, ].copy(), landa)
            print('progress  ', item + 1, '', length)
                
        correction = self.data - matrix
        self.data = correction
        print('Done')
    
    def get_area(self, lower, upper):
        """
        Calculate the area under the curve
        
        Parameters
        ----------
        lower  float
            lower wavenumber axis.
        upper  float
            upper wavenumber axis.
        Returns
        -------
        abu  intensity hyper_object
            area under the curve.
        """
        lower_wave = find_pixel(self.wavenumber, lower)
        upper_wave = find_pixel(self.wavenumber, upper)
        ranging = self.data[:, lower_wave, upper_wave].copy()
        area = np.trapz(ranging, axis=1)
        return area

    def show(self, tittle= 'multi', method=True):
        plt.figure(tittle)
        for count in range(len(self.data)):
            plt.plot(self.wavenumber, self.data[count, ])
        plt.xlabel('Wavenumber (cm-1)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(self.name)
        plt.show()

    def get_data(self):
        return self.data.copy()
    
    def vector_normalization(self):
        """
        Normalization of the data
        Returns
        -------
        None.

        """
        self.data = self.data /np.linalg.norm(self.data, axis=1)[:, None]
    
    def abundance(self, endmembers, constrain):
        """
        It calculates the abudance of the mean spectra onto the spectral map

        Parameters
        ----------
        mean  hyperobject
            hyperobject with the endmembers for abundance calculation.
            
        constrain string
            'NNLS' or 'OLS'
        Returns
        -------
        abu  hyperobject
            concentration of each component.
        """ 
        M = self.data.copy()
        U = endmembers
        if constrain == 'NNLS':
            aux = NNLS(M, U)
        else:
            aux = OLS(M, U)   
        return aux
    def get_wavenumber(self):
        return self.wavenumber.copy()
    
    def rubber_baseline(self):
        """
        Compute rubber band correction (good for converting all values to possitive)
        
        Returns
        -------
        None.

        """
        try:
            aux = self.data.copy()
            x = self.wavenumber
            if len(aux) > 1:
                for count1 in range(len(aux)):
                    y = aux[count1,]
                    baseline = rubberband(x, y)
                    aux[count1,] = aux[count1,] - baseline
                    print(count1+1, '', len(aux))
            else:
                y = aux.copy()
                baseline = rubberband(x, y)
                aux = aux - baseline
            self.data = aux
        except:
            print('Error in rubber_baseline')
