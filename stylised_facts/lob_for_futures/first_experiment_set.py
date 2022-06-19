import time
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
import sys

sys.path.append('/home/ak/Documents/Research/PaperCode/stylised_facts')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import lob_for_futures as lobFut
from lob_for_futures import *
import os
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
import pickle as pkl
import fnmatch
import pickle
import datetime as dt
from datetime import timedelta
import glob
import fathon
from fathon import fathonUtils as fu
import itertools
import pyinform as pyinf
import pingouin as pig
import seaborn as sns
import time
import pickle as pkl

# Lets create a function that shifted your dataframe first before calling the corr().
# Let us break down what we hope to accomplish, and then translate that into code.
# For each hour (row) and for a given feature (column) I would like to find the value
# for that feature N hours prior.
# For each value of N (1-6 in our case) I want to make a new column
# for that feature representing the Nth prior hour's measurement.

elements = '/media/ak/Elements/OrderBookProcessedData'
experimentsLocation = '/media/ak/T7/June4th2022Experiments'
bars = ['volume_bar', 'calendar_bar', 'usd_volume_bar', 'tick_bar']
jb_dict = dict()
standarised_returns = defaultdict(dict)
dataFolder = lobFut.dataFolder
quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]

## THIS NEEEEEEEE DOCUMENTATION

symbols = sorted(os.listdir(elements))
bars = ['volume_bar', 'calendar_bar', 'usd_volume_bar', 'tick_bar']


def jb_calculation(symbolIdx, filesIdx):
    # add location variables. these i have to figure how to abstract away
    print(symbols[symbolIdx])
    procsdSymbolFolder = os.path.join(elements, symbols[symbolIdx])
    print(procsdSymbolFolder)
    files = sorted(os.listdir(procsdSymbolFolder))

    fileLocation = os.path.join(procsdSymbolFolder, files[filesIdx])
    print(fileLocation)

    # pick the various files
    volume_bar_dict = open_pickle_filepath(fileLocation)[bars[0]]
    calendar_bar_dict = open_pickle_filepath(fileLocation)[bars[1]]
    usd_volume_bar_dict = open_pickle_filepath(fileLocation)[bars[2]]
    tick_bar_dict = open_pickle_filepath(fileLocation)[bars[3]]
    # get the dataframes

    volume_bar_df = volume_bar_dict[list(volume_bar_dict.keys())[0]]
    calendar_bar_df = calendar_bar_dict[list(calendar_bar_dict.keys())[0]]
    usd_volume_df = usd_volume_bar_dict[list(usd_volume_bar_dict.keys())[0]]
    tick_bar_df = tick_bar_dict[list(usd_volume_bar_dict.keys())[0]]
    # returns

    vb_ret = returns(volume_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)  # volume
    tb_ret = returns(tick_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)  # tick
    usdvb_ret = returns(usd_volume_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # usd volume
    cb_ret = returns(calendar_bar_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # calendar
    # calculating JB statistic
    jb_value_tick, _ = jarque_bera(tb_ret)
    jb_value_vol, _ = jarque_bera(vb_ret)
    jb_value_dollar, _ = jarque_bera(usdvb_ret)
    jb_value_calendar, _ = jarque_bera(cb_ret)

    jb_test_df = pd.DataFrame(data={'jarque_bera_results': [jb_value_tick,
                                                            jb_value_vol,
                                                            jb_value_dollar,
                                                            jb_value_calendar]
                                    },
                              index=['tick', 'vol', 'dollar', 'calendar'])
    pickle_out_returns = os.path.join(experimentsLocation,
                                      "".join((str(symbols[symbolIdx]), "_" + str(filesIdx) + "_jb_stats.pkl")))
    pickle.dump(jb_test_df, open(pickle_out_returns, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print("produced and saved JB stats for :", symbols[symbolIdx], filesIdx)


def standarised_returns(symbolIdx, filesIdx):
    standarised_returns = defaultdict(dict)
    # add location variables. these i have to figure how to abstract away
    ### all files here ###
    procsdSymbolFolder = os.path.join(elements, symbols[symbolIdx])
    files = sorted(os.listdir(procsdSymbolFolder))
    fileLocation = os.path.join(procsdSymbolFolder, files[filesIdx])

    volume_bar_dict = open_pickle_filepath(fileLocation)[bars[0]]
    calendar_bar_dict = open_pickle_filepath(fileLocation)[bars[1]]
    usd_volume_bar_dict = open_pickle_filepath(fileLocation)[bars[2]]
    tick_bar_dict = open_pickle_filepath(fileLocation)[bars[3]]
    ### all dataframes here ###

    volume_bar_df = volume_bar_dict[list(volume_bar_dict.keys())[0]]
    calendar_bar_df = calendar_bar_dict[list(calendar_bar_dict.keys())[0]]
    usd_volume_df = usd_volume_bar_dict[list(usd_volume_bar_dict.keys())[0]]
    tick_bar_df = tick_bar_dict[list(usd_volume_bar_dict.keys())[0]]

    vb_ret = returns(volume_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)  # volume
    tb_ret = returns(tick_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)  # tick
    usdvb_ret = returns(usd_volume_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # usd volume
    cb_ret = returns(calendar_bar_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # calendar

    tick_standard = (tb_ret - tb_ret.mean()) / tb_ret.std()
    volume_standard = (vb_ret - vb_ret.mean()) / vb_ret.std()
    dollar_standard = (usdvb_ret - usdvb_ret.mean()) / usdvb_ret.std()
    cb_standard = (cb_ret - cb_ret.mean()) / cb_ret.std()
    standarised_returns['tick'][filesIdx] = tick_standard
    standarised_returns['calendar'][filesIdx] = cb_standard
    standarised_returns['volume'][filesIdx] = volume_standard
    standarised_returns['dollar'][filesIdx] = dollar_standard

    pickle_out_returns = os.path.join(experimentsLocation,
                                      "".join((str(symbols[symbolIdx]), "_" + str(filesIdx) + "_standard_returns.pkl")))
    pickle.dump(standarised_returns, open(pickle_out_returns, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return standarised_returns


if __name__ == '__main__':
    a_args = [5] # symbol
    second_arg = [f for f in range(0, 20, 1)] # range of files

    freeze_support()
# produces simple standarised returns and calculates jB statistic
    with Pool() as pool:
        L = pool.starmap(standarised_returns, list(itertools.product(a_args, second_arg)))
        K = pool.starmap(jb_calculation, list(itertools.product(a_args, second_arg)))
