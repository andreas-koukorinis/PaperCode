import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import numpy as np
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sb
###

from tqdm import tqdm

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

import sys

sys.path.insert(0, '/directory/tothe/handshakefile/')
import os

from scipy.stats import norm, ttest_ind

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataFolder = '/media/ak/WorkDrive/Data'
figures_destination = '/home/ak/Documents/Research/Papers/figures/'

import matplotlib as mpl

plt.style.use(os.path.join(mpl.get_configdir(), 'latexstyle.mplstyle'))

import math

import scipy as sp
import scipy
import scipy.optimize as opt
import scipy.stats as stats

import xlwt
import logging


def logger(name, level=logging.INFO):
    tmp_logger = logging.getLogger(name)
    if not tmp_logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(name)s  %(levelname)s %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        tmp_logger.setLevel(level)
        tmp_logger.addHandler(handler)
    return tmp_logger
