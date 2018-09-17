                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
import os
import matplotlib.pyplot as plt
from scipy import signal
import pickle

import itertools as iter
from sklearn import preprocessing as prep
import seaborn as sns


doc_path= ('/home/ak/Documents/Data/')
figures_path= ('/home/ak/Documents/Research/QFPaper/paper/figures/')
data_path_gp = doc_path + 'GPDATA/'
import seaborn as sns
plt.style.use('fivethirtyeight')

#search only for npy files+make a list --> will use this later
npy_files=[s for s in os.listdir(doc_path+'mySYNT/features/') if s.endswith('.npy')]

#load bot matrix and vector valued
ksi_metrics =np.load(data_path_gp+'ksi_metrics.npy').item()
im_metrics =np.load(data_path_gp+'im_metrics.npy').item()
fischer_polar =np.load(data_path_gp+'fischer_polar.npy')

c=norm_spectral_cs[1200]
plt.plot(a,linewidth=0.4)
plt.plot(b, color='r',linewidth=0.5,linestyle='dashdot')
plt.plot(c, color='b',linewidth=0.5,linestyle=':')
plt.xlabel('#copies of sequences')
plt.legend()
plt.savefig(figures_path+'determ_copies.png')   # save the figure to file
plt.close()    # close the figure

#fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

for j in range(1,14):
    x_series =norm_spectral_cs[j*100]
    plt.plot(x_series, label=str(j*100), linewidth=0.4,linestyle='-')
    _keys=im_metrics.keys()
    _keys #names are the same for all files
    #deal with ksi-first
    ksi_spectral=pd.DataFrame(ksi_metrics[_keys[0]])
    ksi_determ =pd.DataFrame(ksi_metrics[_keys[1]])
    ksi_trace=pd.DataFrame(ksi_metrics[_keys[2]])

    no_paths =ksi_spectral.shape[0]
    seq_length=ksi_spectral.shape[1]
    print "the number of copies (rows):", no_paths
    print "the number of sequences (columns):", seq_length

    #summing across copies (each column is a running sum)
    _df_= ksi_trace
    spectral_cs =_df_.cumsum()
    denominator=spectral_cs.index.values+1
    norm_spectral_cs =spectral_cs.divide(denominator, axis='rows')
    norm_spectral_average = norm_spectral_cs.mean(axis=1) #calculate average
    _std=norm_spectral_cs.std(axis=1) #calculate standard-deviation

    norm_spectral_average.plot()

    norm_spectral_cs['index']=norm_spectral_cs.index

    x = np.asarray(norm_spectral_cs.index)
    y = np.asarray(norm_spectral_average)
    error = np.asarray(_std)

    plt.plot(x, y, 'k-',linewidth=0.7)
    plt.fill_between(x, y-error, y+error, alpha=0.6, edgecolor='#1B2ACD', facecolor='#089FFF',
                     linewidth=0.1, linestyle='dashdot', antialiased=False)
    plt.title('Average Sequence')
    plt.legend()
    plt.savefig(figures_path+'determ_average_sequence.png')   # save the figure to file
    plt.close()    # close the figure

    a=norm_spectral_cs[900]
    b=norm_spectral_cs[1100]

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Ksi Determinant', fontsize='small')
ymin=0.85
ymax=0.95
plt.ylim(ymin, ymax)
plt.show()
plt.savefig(figures_path+'determ_seq.png')   # save the figure to file
plt.close()

sns.distplot(norm_spectral_average,hist=True)
plt.title('Distribution for Ksi-Normalised Spectral Average',fontsize='small')
plt.savefig(figures_path+'ksi_determinant_distribution.png')   # save the figure to file
plt.close()


