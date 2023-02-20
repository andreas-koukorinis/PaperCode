import datetime as dt
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
plt.style.use(os.path.join(mpl.get_configdir(), 'latexstyle3.mplstyle'))
# for debugging
import pdb

import pickle
data ='/media/ak/OS/Data'
npyfiles = os.path.join(data, 'npyfiles')
im_metrics = os.path.join(npyfiles, 'im_metrics.npy')

print(np.load(im_metrics, allow_pickle=True, encoding = 'latin1').item())