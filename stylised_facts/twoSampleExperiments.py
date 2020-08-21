import twosample as twosample

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import numpy as np
from scipy.special import gamma
import time

try:
    from tqdm import tqdm_notebooks as tqdm
except ImportError:
    tqdm = lambda x: x
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import stylised_facts_data_utilities.gpyNARX as gpyNARX
import stylised_facts_data_utilities.longtaildistr as longtail
import stylised_facts_data_utilities.createLOB as createLOB

from scipy import stats

# locations

etfDir = '/media/ak/My Passport1/MarketData/ETF_Levels/'
etfSymbols = os.listdir(etfDir)  # etf symbols

if __name__ == '__main__':
    idx = 3  # choice of ETF
    etfSymbolFiles = os.path.join(etfDir, etfSymbols[idx])  # picking the idx specific symbol
    listETFSymbolFiles = os.listdir(etfSymbolFiles)  # all the files of the specific ETF symbol

    combinedETFfiles = dict()
    for fileIdx, _ in enumerate(os.listdir(etfSymbolFiles)):
        etfFileDate = os.listdir(etfSymbolFiles)[fileIdx]
        fileLoc = os.path.join(etfSymbolFiles, os.listdir(etfSymbolFiles)[fileIdx])
        combinedETFfiles[etfFileDate.split('.')[0]] = createLOB.formatETFlob(
            pd.read_csv(fileLoc))  # append the dictionary with the date

# TODO: store the combined file

    etfDateKeys = list(combinedETFfiles.keys())
