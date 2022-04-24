import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import lob_for_futures as lobFut
from lob_for_futures import *
import os
import matplotlib.pyplot as plt

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
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
import time
import matplotlib.ticker as mtick
dataFolder = lobFut.dataFolder
quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]


# Lets create a function that shifted your dataframe first before calling the corr().
# Let us break down what we hope to accomplish, and then translate that into code.
# For each hour (row) and for a given feature (column) I would like to find the value
# for that feature N hours prior.
# For each value of N (1-6 in our case) I want to make a new column
# for that feature representing the Nth prior hour's measurement.

def df_derived_by_shift(df, lag=0, NON_DER=[]):
    df = df.copy()
    if not lag:
        return df
    cols = {}
    for i in range(1, lag + 1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i += 1
        df = pd.concat([df, dfn], axis=1)
    return df


def get_test_stats(bar_types, bar_returns, test_func, *args, **kwds):
    dct = {bar: (int(bar_ret.shape[0]), test_func(bar_ret, *args, **kwds))
           for bar, bar_ret in zip(bar_types, bar_returns)}
    df = (pd.DataFrame.from_dict(dct)
          .rename(index={0: 'sample_size', 1: f'{test_func.__name__}_stat'})
          .T)
    return df

if __name__ == '__main__':

    symbolIdx = 0

    symbols = [f.split("_")[0] for f in quotes]
#
    symbol = sorted(symbols)[symbolIdx]
    print(symbol)
    quotesFileCh = os.path.join(dataFolder, quotes[symbolIdx])
    tradesFileCh = os.path.join(dataFolder, trades[symbolIdx])
#
    # # get common Dates
    quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])
    tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])
    intersectionDates = list(set(quotesDates).intersection(tradesDates))

    # params i need for fathon
    winSizes = fu.linRangeByStep(5, 50)
    revSeg = True
    qs = np.arange(-3, 4, 0.1)
    polOrd = 3
    # load all teh dataframes at once with this
    start = time.time()

    # go through all the dataframes and take out n, f and h, h_ intercept
    # somehow one list comprehension blows up in memory

    h_dict = defaultdict(dict)

    trades_cols = ['size', 'time', 'type', 'value']

    # params for the clocks

    calendar_resample_freq = "300S"
    trade_volume_width = 100
    ticks_width = 100
    usd_volume_width = 100
    dates_choice = intersectionDates[0:5]

    testClass = DataLoader(data_location=dataFolder,
                           symbol=symbol,
                           dates=dates_choice,
                           use_columns=trades_cols,
                           calendar_resample=calendar_resample_freq,
                           trade_volume_width=trade_volume_width,
                           ticks_width=ticks_width,
                           usd_volume_width=usd_volume_width)

    hash_of_file = "_".join(
        (str(symbol), "volume_width", str(trade_volume_width), "calendar_resample", str(calendar_resample_freq)))

    # load data
    input_dict = testClass.load_and_format_data()

    tick_bar_dfs = []
    volume_bar_dfs = []
    usd_volume_bar_dfs = []
    calendar_bar_dfs = []
    dates = list(input_dict.keys())

    bar_returns = dict()
    bars_dicts = defaultdict(dict)
    end = time.time()
    print(end - start) # just so i can keep track how long it takes to print
    print(dates)
    start2 = time.time()
    for date in dates:
        df = testClass.load_and_format_data()[str(date)]
        input_dict = testClass.get_bars(df)
        tick_bar_df = testClass.get_concat_data(testClass._bars_dict)['tick_bars']
        volume_bar_df = testClass.get_concat_data(testClass._bars_dict)['volume_bars']
        usd_volume_bar_df = testClass.get_concat_data(testClass._bars_dict)['usd_volume_bars']
        calendar_bar_df = testClass.get_concat_data(testClass._bars_dict)['calendar_bars']
        vr = returns(volume_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)  # volume
        tr = returns(tick_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)  # tick
        dr = returns(usd_volume_bar_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # usd volume
        df_ret = returns(calendar_bar_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # calendar
        bar_returns[date] = {'tick': tr,
                             'volume': vr,
                             'dollar': dr,
                             'calendar': df_ret}
        bars_dicts[date]['tick'] = tick_bar_df
        bars_dicts[date]['volume'] = volume_bar_df
        bars_dicts[date]['calendar'] = calendar_bar_df
        bars_dicts[date]['dollar'] = usd_volume_bar_df
    end2 = time.time()
    print(end2 - start2)