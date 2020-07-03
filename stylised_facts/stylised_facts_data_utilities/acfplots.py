import mdfda as mdf
# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import numpy as np
from numpy import cumsum, polyfit, polyval, mean, sqrt, arange, zeros, nan, log2
from numpy.lib.stride_tricks import as_strided as ast
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
plt.style.use('bmh')

#plt.rcParams['font.family'] = 'DejaVu Sans Mono'
#plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'
#plt.rcParams['figure.figsize'] = 10,7
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)

# import util libs

from tqdm import tqdm, tqdm_notebook
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 777
####

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import numpy as np

import time
import seaborn as sb
###
from MFDFA import fgn
# Plot settings
from matplotlib import pyplot as plt
from matplotlib import cm

import matplotlib.gridspec as gridspec

RANDOM_STATE = 777

## Plot Settings ##
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
###

# Calculate autocorrelation for n lags
def figAutocorrel(ticks_adj,title = 'autocorrelsymbol',xlineLevel=80):
    from matplotlib.ticker import AutoMinorLocator
    size=15
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20,8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size*0.75,
              'ytick.labelsize': size*0.75,
              'axes.titlepad': 25}
    fig, ax1 = plt.subplots(figsize=(16,9))
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    #dictionary
    autoCorrelDict ={}

    # Set up the second axis
    ax2 = ax1.twiny()
    ax1.set_facecolor('xkcd:white')
    for i in np.arange( 1, 500 ):
        ax1.bar( i, ticks_adj.autocorr( lag = i ), color = 'blue' )
    ax1.set_title( 'Autocorrelation by lag' )
    ax1.set_ylabel( 'Autocorrelation' )
    ax1.set_xlabel( 'Lag' )
    xtick_labels = [r'$0$', r'$\tau$'] + [r'${}\tau$'.format(k) for k in range(2,ntau)]
    ax2.set_xticklabels(xtick_labels)
    # Initial value of y at t=0, lifetime in s
    N, tau = 1,15
    # Maximum time to consider (s)
    tmax = 500
    # A suitable grid of time points, and the exponential decay itself
    t = np.linspace(0, tmax, 400)
    y = N * np.exp(-t/tau)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax1.plot(t, y, color='black', linestyle ='--')
#     axvline(x=.5, ymin=0.25, ymax=0.75)
    ax1.axvline(x=xlineLevel,ymin=0.0, ymax=0.8, ls='--', alpha=0.7, color='#334f8d')
    ax1.text(xlineLevel,0.85,'lag '+str(xlineLevel),rotation=90, fontsize=12)
    plt.savefig('/home/ak/Documents/Research/Papers/figures/'+title+'.png', dpi=150)
    plt.show()