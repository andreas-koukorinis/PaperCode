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
import pingouin as pig
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
import time

dataFolder = lobFut.dataFolder
quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]

# symbolIdx = 0

symbols = [f.split("_")[0] for f in quotes]
#
# symbol = sorted(symbols)[symbolIdx]
# print(symbol)
# quotesFileCh = os.path.join(dataFolder, quotes[symbolIdx])
# tradesFileCh = os.path.join(dataFolder, trades[symbolIdx])
#
# # get common Dates
# quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])
# tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])
# intersectionDates = list(set(quotesDates).intersection(tradesDates))

trades_cols = ['size', 'time', 'type', 'value']
experimentsDestination = '/media/ak/T7/MFDFA Experiments'

if __name__ == '__main__':
    # params i need for fathon
    winSizes = fu.linRangeByStep(5, 50)
    revSeg = True
    qs = np.arange(-3, 4, 0.1)
    polOrd = 3
    # load all teh dataframes at once with this
    start = time.time()
    print(start)
    # go through all the dataframes and take out n, f and h, h_ intercept
    # somehow one list comprehension blows up in memory

    h_dict = defaultdict(dict)

    symbolIdx = 2 # symbol 3 has problems

    symbol = sorted(symbols)[symbolIdx]
    print(symbol)
    quotesFileCh = os.path.join(dataFolder, quotes[symbolIdx])
    tradesFileCh = os.path.join(dataFolder, trades[symbolIdx])

    # get common Dates
    quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])
    tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])
    intersectionDates = list(set(quotesDates).intersection(tradesDates))

    trades_cols = ['size', 'time', 'type', 'value']

    # params for the clocks

    calendar_resample_freq = "200S"
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
    # -storage locations! -
    mfdfa_n_F_dict = defaultdict(dict)
    mfdfa_H_dict = defaultdict(dict)
    mfdfa_tau_dict = defaultdict(dict)
    mfdfa_alpha_spect = defaultdict(dict)
    dict_dfa = defaultdict(dict)
    h_dict = defaultdict(dict)

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

    for j, i in itertools.product(['tick', 'volume', 'dollar', 'calendar'], dates):
        data = (bar_returns[i][j])
        a = fu.toAggregated(np.asanyarray(data))
        # MFDFA Computations
        pymfdfa = fathon.MFDFA(a)
        n, F = pymfdfa.computeFlucVec(winSizes, qs, revSeg=revSeg, polOrd=polOrd)

        mfdfa_n_F_dict[j][i] = dict(zip(n, F))
        # dictionary to match all the n and F values this could be
        # more efficient

        # get the list values of H and intercept
        list_H, list_H_intercept = pymfdfa.fitFlucVec()  # same for H values
        mfdfa_H_dict[j][i] = [list_H, list_H_intercept]

        # get the mass exponents
        tau = pymfdfa.computeMassExponents()
        mfdfa_tau_dict[j][i] = tau

        # get the multi-fractal spectrum
        alpha, mfSpect = pymfdfa.computeMultifractalSpectrum()
        mfdfa_alpha_spect[j][i] = [mfSpect, alpha]
        pydfa = fathon.DFA(a)
        n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
        dict_dfa[j][i] = dict(zip(n, F))  # dictionary to match all the n and F values

        H, H_intercept = pydfa.fitFlucVec()  # same for H values
        h_dict[j][i] = [H, H_intercept]

        print('for date', i, ' and bar type', j, 'you get', H)

    # everything below here is to store
    # saving DFA files first

    # -1-

    pickle_out_filename = os.path.join(experimentsDestination, symbol, 'DFA_' +
                                       'n_f_values' + "volume_width" + str(trade_volume_width)
                                       + "calendar_resample" + str(calendar_resample_freq) +
                                       '_bar_' + str(i) + '_' + str(j) + '.pkl')

    pickle.dump(dict_dfa, open(pickle_out_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('just saved: ', pickle_out_filename)

    # -2- saving the DFA n_f Values
    pickle_out_filename_two = os.path.join(experimentsDestination, symbol, 'DFA_' +
                                           'h_dict_values' + "volume_width" + str(trade_volume_width)
                                           + "calendar_resample" + str(calendar_resample_freq) +
                                           '_bar_' +  str(i) + '_' + str(j) + '.pkl')



    pickle.dump(h_dict, open(pickle_out_filename_two, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # saving MFDFA files next

    # -3-  Saving the mfdfa _n_f_values
    pickle_out_filename_three = os.path.join(experimentsDestination, symbol, 'MDFA_' +
                                             'n_f_values' + "volume_width" + str(trade_volume_width)
                                             + "calendar_resample" + str(calendar_resample_freq) +
                                             '_bar_' + str(i) + '_' + str(j) + '.pkl')

    pickle.dump(mfdfa_n_F_dict, open(pickle_out_filename_three, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # saving mfdfa tau -mass exponent

    print('just saved: ', pickle_out_filename_three)
    # -4- Saving the tau-dict
    pickle_out_filename_four = os.path.join(experimentsDestination, symbol, 'MDFA_' +
                                            'TAU_DICT' + "volume_width" + str(trade_volume_width)
                                            + "calendar_resample" + str(calendar_resample_freq) +
                                            '_bar_' + str(i) + '_' + str(j) + '.pkl')

    pickle.dump(mfdfa_tau_dict, open(pickle_out_filename_four, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('just saved: ', pickle_out_filename_four)

    # -5 - Saving the alpha_spect
    pickle_out_filename_five = os.path.join(experimentsDestination, symbol, 'MDFA_' +
                                            'alpha_spect' + "volume_width" + str(trade_volume_width)
                                            + "calendar_resample" + str(calendar_resample_freq) +
                                            '_bar_' + str(i) + '_' + str(j) + '.pkl')

    pickle.dump(mfdfa_alpha_spect, open(pickle_out_filename_five, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('just saved: ', pickle_out_filename_five)

    # -6- saving mfdfa - hdict

    pickle_out_filename_six = os.path.join(experimentsDestination, symbol, 'MDFA_' +
                                           'H_dict' + "volume_width" + str(trade_volume_width)
                                           + "calendar_resample" + str(calendar_resample_freq) +
                                           '_bar_' + str(i) + '_' + str(j) + '.pkl')

    pickle.dump(mfdfa_H_dict, open(pickle_out_filename_six, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('just saved: ', pickle_out_filename_six)
