# This is a sample Python script.

import os

import numpy as np
import pandas as pd
import os
import pandas as pd
import pickle
import numpy as np
# from shogun import *
import numpy as np
import matplotlib.pyplot as plt


import random

import time
import ksd
import mmd
import tf_ops

import matplotlib as mpl

# import seaborn as sns
# sns.set(context='notebook')

from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm  # if you're in JupyterLab/etc and this doesn't work well

import functools

import numpy as np
import sklearn
from sklearn import model_selection


import ds3_support as ds3s
from ds3_support import as_tensors, LazyKernel, pil_grid
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import stylised_facts.stylised_facts_data_utilities.createLOB as createLOB

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# getting all the data in
ActivityCLockData= '/media/ak/My Passport/Experiment Data/ActivityClockData/'
OHLCData= '/media/ak/My Passport/Experiment Data/OHLCData/'
folderList = os.listdir(ActivityCLockData)
OHLCDataList = os.listdir(OHLCData)
ActivityClockDataList = os.listdir(ActivityCLockData)

symbols =['FB1','JB1','FV1','G_1']
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    symbol1 = symbols[1]
    Symbol1ClocksData = np.sort([s for s in os.listdir(ActivityCLockData) if 'Clocks' in s and (str(symbol1)) in s])
    print(Symbol1ClocksData)