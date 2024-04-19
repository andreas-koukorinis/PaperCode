import time
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
import sys

sys.path.append('/')
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


elements = '/media/ak/Elements/OrderBookProcessedData'
experimentsLocation = '/media/ak/T7/June4th2022Experiments'
bars = ['volume_bar', 'calendar_bar', 'usd_volume_bar', 'tick_bar']
# plt.style.use(os.path.join('/home/ak/.config/matplotlib', 'latexstyle.mplstyle'))
figures_destination = '/home/ak/Documents/Research/Papers/figures/'

jb_dict = dict()
standarised_returns = defaultdict(dict)
# dataFolder = lobFut.dataFolder
# quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
# trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]


def produce_dict(user_dict):
    columns_values = list(user_dict.keys())
    _df = pd.DataFrame.from_dict({(i, j): user_dict[i][j]
                                  for i in user_dict.keys()
                                  for j in user_dict[i].keys()},
                                 orient='index')
    return_df = _df.reset_index().drop(columns=['level_0', 'level_1'])
    return return_df


if __name__ == '__main__':
    files = [f for f in os.listdir('/media/ak/T7/June4th2022Experiments/') if str('_standard_returns') in f]
    symbol_ = 'JB1'
    print(files)
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
        loaded_returns = open_pickle_filepath(file_path)
        # print(list(loaded_returns['tick'].keys())[0])

        index_item = (list(loaded_returns['tick'].keys())[0])
        print(index_item)
        tick_dict[file_index + 1] = (pd.DataFrame(loaded_returns['tick'][index_item].values))
        usd_dict[file_index + 1] = (pd.DataFrame(loaded_returns['dollar'][index_item].values))
        calendar_dict[file_index + 1] = (pd.DataFrame(loaded_returns['calendar'][index_item].values))
        volm_dict[file_index + 1] = (pd.DataFrame(loaded_returns['volume'][index_item].values))

    # this is extremely rudimentary

    tick_ = produce_dict(tick_dict).T.median(axis=1)
    usd_ = produce_dict(usd_dict).T.median(axis=1)
    volm_ = produce_dict(volm_dict).T.median(axis=1)
    calendar_ = produce_dict(calendar_dict).T.median(axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.kdeplot(tick_,
                ax=ax,
                label="Tick",
                bw=0.25,
                color='darkblue')
    sns.kdeplot(usd_,
                ax=ax,
                label="USD Volume",
                bw=1.55, linewidth=2.25,
                color='purple', linestyle='-')

    sns.kdeplot(volm_,
                ax=ax,
                label="Volume",
                bw=0.55, linewidth=1.25,
                color='red', linestyle='-')
    sns.kdeplot(calendar_,
                ax=ax,
                label="Calendar",
                bw=0.55, linewidth=2.25,
                color='grey', linestyle=':')

    sns.kdeplot(np.random.normal(size=len(usd_)),
                ax=ax,
                label="Normal",
                color='darkgreen',
                linewidth=3,
                linestyle="--")
    plt.legend()

    _ = plt.xlabel('Returns', fontsize=19)
    _ = plt.ylabel('Density', fontsize=19)
    _ = plt.xticks(fontsize=15, rotation=45)
    _ = plt.yticks(fontsize=15)

    # plt.title(str(symbol))
    file_name = str(symbol_) + 'median_multiclocks_density_plot.png'
    plt.savefig(os.path.join(figures_destination, file_name))
    plt.legend()
    plt.show()
