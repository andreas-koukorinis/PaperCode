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

# symbolIdx = 0

symbols = [f.split("_")[0] for f in quotes]

from multiprocessing import Pool
import pickle

trades_cols = ['size', 'time', 'type', 'value']
experimentsDestination = '/media/ak/T7/MFDFA Experiments'
#plt.style.use(os.path.join('/home/ak/.config/matplotlib', 'latexstyle.mplstyle'))
informationClockFolder = '/media/ak/T7/FuturesDataSemiProcessed'


def produce_info_clock_files(date):
    bar_returns = dict()
    bars_dicts = defaultdict(dict)
    informationClockFolder = '/media/ak/T7/FuturesDataSemiProcessed'

    df = testClass.load_and_format_data()[str(date)]

    input_dict = testClass.get_bars(df)
    calendar_bar_df = (testClass.get_concat_data(testClass._bars_dict)['calendar_bars'])
    tick_bar_df = testClass.get_concat_data(testClass._bars_dict)['tick_bars']
    volume_bar_df = (testClass.get_concat_data(testClass._bars_dict)['volume_bars'])
    usd_volume_bar_df = testClass.get_concat_data(testClass._bars_dict)['usd_volume_bars']
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
    print(bar_returns)
    symbolFolderToStore = os.path.join(informationClockFolder, symbol)

    #  store all values
    pickle_out_filename = os.path.join(symbolFolderToStore, str(date) + '.pkl')
    pickle_out_filename_returns = os.path.join(symbolFolderToStore, 'Returns_' + str(date) + '.pkl')

    pickle.dump(bars_dicts, open(pickle_out_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(bar_returns, open(pickle_out_filename_returns, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('just saved: ', pickle_out_filename)

    # return bar_returns, bars_dicts


if __name__ == '__main__':

    symbolIdx = 15

    symbol = str('VXX')
    print(symbol)

    symbolFolderToStore = os.path.join(informationClockFolder, symbol)
    print(symbolFolderToStore)
    number_of_workers = 5

    quotesFileCh = os.path.join(dataFolder, str('VXX_US_Equity_quotes'))
    tradesFileCh = os.path.join(dataFolder, str('VXX_US_Equity_trades'))

    # get common Dates
    quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])
    tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])
    intersectionDates = list(set(quotesDates).intersection(tradesDates))

    trades_cols = ['size', 'time', 'type', 'value']

    # params for the clocks

    calendar_resample_freq = "300S"
    trade_volume_width = 100
    ticks_width = 100
    usd_volume_width = 100
    dates_choice = intersectionDates

    testClass = DataLoader(data_location=dataFolder,
                           symbol=symbol,
                           dates=dates_choice,
                           use_columns=trades_cols,
                           calendar_resample=calendar_resample_freq,
                           trade_volume_width=trade_volume_width,
                           ticks_width=ticks_width,
                           usd_volume_width=usd_volume_width)

    # load data
    input_dict = testClass.load_and_format_data()

    # with Pool(number_of_workers) as p:
    try:
        with Pool(number_of_workers) as p:
            print(p.map(produce_info_clock_files, intersectionDates))
    except KeyError:
        pass
