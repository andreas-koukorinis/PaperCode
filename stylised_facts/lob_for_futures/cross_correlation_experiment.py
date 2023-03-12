import pandas as pd
import numpy as np
import sys
import multiprocessing
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
import pickle as pkl
import fnmatch
import lob_for_futures as lobFut
from lob_for_futures import *
# extra imports
from collections import OrderedDict
from itertools import zip_longest
import datetime as dt
from datetime import timedelta
import time
import glob
import fathon
from fathon import fathonUtils as fu

winSizes = fu.linRangeByStep(10, 1000, step=20)
polOrd = 1
import time

import matplotlib as mpl
from multiprocessing import Pool, freeze_support, Manager

import itertools

# colormap = plt.cm.RdBu
# plt.style.use(os.path.join(mpl.get_configdir(), 'latexstyle.mplstyle'))
from collections import defaultdict

dataFolder = lobFut.dataFolder
t7 = lobFut.t7folder
expInputFiles = os.path.join(lobFut.augExpertiments, 'ExperimentInputFiles')
HiLoData = os.path.join(lobFut.augExpertiments, 'HiLoData')
plt.style.use(os.path.join('/home/ak/.config/matplotlib', 'latexstyle3.mplstyle'))
symbols = ['RX1', 'FB1', 'JB1', 'G_1', 'FV1', 'TY1', 'TU1', 'DU1', 'YM1', 'XM1', 'US1', 'OE1', 'KE1']

# laptop data folder
laptopDataFolder = os.path.join('/media/ak/OS', 'Data')
expFiles = os.path.join(laptopDataFolder, 'August11th2022Experiments')

destination = os.path.join(laptopDataFolder, 'AthensSoloTripFeb2023')
figuresDestination = '/home/ak/Documents/Papers/StylisedFactsPaper/figures/'
# n: Array of window's sizes used for the computation.
n = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
     39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
     68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]


class CrossCorrel():
    def __init__(self, path, symbols_list, idx, bar_choice):
        """

        :param path: which path to read from for the data
        :param symbols_list:
        :param idx:
        :param bar_choice:
        """

        self._idx = idx  # this index is for the symbol
        self._symbols = symbols_list
        self._bar = bar_choice
        self._symbol = self._symbols[self._idx]

        self._symbolFilepath = os.path.join(path, str(self._symbol))

        self._listOfFiles = os.listdir(self._symbolFilepath)

        self._micro_variables = ['arrival_rates', 'gk_vol', 'median_traded_volume', 'micro_price_change']

    def get_data_and_path(self):
        # , fileIdx, symbols, path
        """

        :param symbolIdx: symbol index from list
        :param symbols: list of symbols
        :param bar: bar for information clock
        :param path: which path to read from
        :return: list of files, and path to those files
        """
        # print('data for symbol', self._symbol)
        filesBarSymbol = [f for f in self._listOfFiles if str(self._bar) in f]

        return filesBarSymbol, self._symbolFilepath

    def get_all_data_from_file(self, fileIdx):
        """
        fileIdx = index file to get unpickled
        """
        filesBarSymbol, symbolFilepath = self.get_data_and_path()
        fileToGet = os.path.join(symbolFilepath, filesBarSymbol[fileIdx])
        return pd.read_pickle(fileToGet)

    def get_microvar_data(self, fileIdx, var):
        """

        :param fileIdx: index of file in list
        :param var: which micro-structure market variable -string
        :return: returns the microstructure variable
        """
        pklDict = self.get_all_data_from_file(fileIdx)
        return pklDict[str(var)]

    def get_all_microvar_data(self, fileIdx):
        """

        :param fileIdx: index in the list- position of file in the list
        :return: all the microstructure data
        """
        pkl_dict = self.get_all_data_from_file(fileIdx)
        pkl_dict_keys = sorted(list(pkl_dict.keys()))  # get out all the keys- but its really bars

        # ----micro structure vars-----#

        gk_vol = pkl_dict['gk_vol']  # get the vol
        median_traded_volume = pkl_dict['median_traded_volume']  # get the volume
        arrival_rates = pkl_dict['arrival_rates']  # get arrival rates
        micro_price_change = pkl_dict['micro_price_change']  # get micro price change

        # ---- start getting all the mfdfa variables ----#

        tau_dict = pkl_dict[str(self._bar)]['tau']  # tau
        alpha = pkl_dict[str(self._bar)]['alpha']  # alpha
        mfSpect = pkl_dict[str(self._bar)]['mfSpect']  # mfSpect
        n_F_dict = pkl_dict[str(self._bar)]['n_F']  # n_F
        list_H_dict = pkl_dict[str(self._bar)]['list_H']  # list_H
        list_H_intercept = pkl_dict[str(self._bar)]['list_H_intercept']  # intercept

        return gk_vol, median_traded_volume, arrival_rates, micro_price_change, tau_dict, alpha, mfSpect, n_F_dict, list_H_dict, list_H_intercept

    @staticmethod
    def compute_n_Rho(var_a, var_b, idx, winSizes, polOrd):
        """
        var_a = first variable for the dCCa rho
        var_b = second variable for the dCCa rho
        idx = index in for which "day to pick"
        """
        a = fu.toAggregated(var_a[idx])
        b = fu.toAggregated(var_b[idx])
        try:
            pydcca = fathon.DCCA(a, b)
            n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd)
            n, rho = pydcca.computeRho(winSizes, polOrd=polOrd)
        except ZeroDivisionError:
            pass
        return n, rho

    @staticmethod
    def compute_n_F_dcca(var_a, var_b, idx, winSizes, polOrd):
        """
        var_a = first variable for the dCCa rho
        var_b = second variable for the dCCa rho
        idx = index in for which "day to pick"
        """
        a = fu.toAggregated(var_a[idx])
        b = fu.toAggregated(var_b[idx])
        pydcca = fathon.DCCA(a, b)
        n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd)
        return n, F

    @staticmethod
    def compute_H_H_intc_dcca(var_a, var_b, idx, winSizes, polOrd):
        """
        var_a = first variable for the dCCa rho
        var_b = second variable for the dCCa rho
        idx = index in for which "day to pick"
        """
        a = fu.toAggregated(var_a[idx])
        b = fu.toAggregated(var_b[idx])
        pydcca = fathon.DCCA(a, b)
        n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd)

        H, H_intercept = pydcca.fitFlucVec()
        return n, F, H, H_intercept


if __name__ == '__main__':
    symbols = ['RX1', 'FB1', 'JB1', 'G_1', 'FV1', 'TY1', 'TU1', 'DU1', 'YM1', 'XM1', 'US1', 'OE1', 'KE1']
    winSizes = fu.linRangeByStep(10, 1000, step=20)
    polOrd = 1
    import time

    micro_variables = ['arrival_rates', 'gk_vol', 'median_traded_volume', 'micro_price_change']
    # self, path, symbols, idx, bar, use_var
    symbolsIdx = 5
    bar = 'volume'
    cc2 = CrossCorrel(laptopDataFolder, symbols, symbolsIdx, str(bar))
    files, filesPath = cc2.get_data_and_path()
    range_to_use = range(0, 100)
    arrivals_list = {f: cc2.get_microvar_data(fileIdx=f, var='arrival_rates') for f in range_to_use}
    median_volumes_list = {f: cc2.get_microvar_data(fileIdx=f, var='median_traded_volume') for f in range_to_use}
    arrivalsDF = pd.DataFrame(arrivals_list).fillna(0)
    medianVolumesDF = pd.DataFrame(median_volumes_list).fillna(0)


    def fn_for_rho(idx):
        rhoDict = dict()
        n = []
        #  = cc2.compute_n_Rho(arrivalsDF, medianVolumesDF, idx, winSizes, polOrd)
        n, rhoDict[idx] = cc2.compute_n_Rho(arrivalsDF, medianVolumesDF, idx, winSizes, polOrd)
        rhoDict['n'] = n
        file_name = "_".join(('RhoDict_', str(symbols[symbolsIdx]), str(idx), bar, '.pkl'))
        pickle_out_filename = os.path.join(expInputFiles, file_name)
        pickle.dump(rhoDict, open(pickle_out_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        return rhoDict


    def fn_for_H_H_dcca(idx):
        h_dcca = dict()
        #  = cc2.compute_n_Rho(arrivalsDF, medianVolumesDF, idx, winSizes, polOrd)
        h_dcca['n'], h_dcca['F'], h_dcca['H'], h_dcca['H_intercept'] = cc2.compute_H_H_intc_dcca(arrivalsDF, medianVolumesDF, idx, winSizes, polOrd)
        file_name = "_".join(('HurstDict_', str(symbols[symbolsIdx]),str(idx), bar, '.pkl'))
        pickle_out_filename = os.path.join(expInputFiles, file_name)
        pickle.dump(h_dcca, open(pickle_out_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        return h_dcca

    tic = time.perf_counter()
    # fn_for_rho(1)
    with Pool(5) as p:
        print(p.map(fn_for_H_H_dcca, [f for f in range(0, 20)]))
    toc = time.perf_counter()
    print("elapsed time:", (toc - tic))
