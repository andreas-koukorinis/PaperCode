import time
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
import sys

sys.path.append('/home/ak/Documents/Research/PaperCode/stylised_facts')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import lob_for_futures as lobFut
from lob_for_futures import *

import matplotlib.pyplot as plt

from dateutil.parser import parse
from pandas.tseries.offsets import BDay
import pickle as pkl
import fnmatch
import pickle
import datetime as dt
from datetime import timedelta
import glob

from fathon import fathonUtils as fu
import itertools
# import pyinform as pyinf
# import pingouin as pig
import time
import pickle as pkl
import seaborn as sns
#########
# read the standard return files and perform mmd- comparison between them
# then produce the various plots that are needed
if __name__ == '__main__':
    elements = lobFut.elements
    symbols = lobFut.symbols
    bars = lobFut.bars
    experimentsLocation = lobFut.experimentsLocation
    files = os.listdir(experimentsLocation) # stored data from previous experiments
    symbol_ = 'JB1'
    type_file = '_standard_returns'
    type_files = [f for f in files if str(type_file) in f]
    symbol_type_files = [g for g in type_files if str(symbol_) in g]
    print(symbol_type_files)

    tick_dict = dict()
    usd_dict = dict()
    calendar_dict = dict()
    volm_dict = dict()

    for file_index, _ in enumerate(symbol_type_files):
        # print(file_index)
        file_path = os.path.join(experimentsLocation, symbol_type_files[file_index])
        # print(file_path)
        loaded_returns = lobFut.open_pickle_filepath(file_path)
        # print(list(loaded_returns['tick'].keys())[0])

        index_item = (list(loaded_returns['tick'].keys())[0])
        print(index_item)
        tick_dict[file_index + 1] = (pd.DataFrame(loaded_returns['tick'][index_item].values))
        usd_dict[file_index + 1] = (pd.DataFrame(loaded_returns['dollar'][index_item].values))
        calendar_dict[file_index + 1] = (pd.DataFrame(loaded_returns['calendar'][index_item].values))
        volm_dict[file_index + 1] = (pd.DataFrame(loaded_returns['volume'][index_item].values))

        print(volm_dict[1])
        # pick a symbol
        # pick an index (starting)
        # pick a clock
        # load and do mmd-test



