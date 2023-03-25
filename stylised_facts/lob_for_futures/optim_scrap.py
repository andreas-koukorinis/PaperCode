import pandas as pd
import numpy as np
import sys
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
import pickle as pkl
import fnmatch
from collections import OrderedDict
from itertools import zip_longest
import datetime as dt
from datetime import timedelta
import time
import glob
import fathon
from fathon import fathonUtils as fu
import itertools
from multiprocessing import Pool, freeze_support, Manager, cpu_count
from collections import defaultdict
import lob_for_futures as lobFut
import pickle
# Define constants
winSizes = fu.linRangeByStep(10, 1000, step=20)
polOrd = 1
# n: Dictionary of window's sizes used for the computation.
n = {i: i for i in range(10, 80)}

# Define paths and folders
dataFolder = lobFut.dataFolder
t7 = lobFut.t7folder
expInputFiles = os.path.join(lobFut.augExpertiments, 'ExperimentInputFiles')
HiLoData = os.path.join(lobFut.augExpertiments, 'HiLoData')
symbols = ['RX1', 'FB1', 'JB1', 'G_1', 'FV1', 'TY1', 'TU1', 'DU1', 'YM1', 'XM1', 'US1', 'OE1', 'KE1']
t7DataFolder = os.path.join(t7, 'August11th2022Experiments/ExperimentOne/')
laptopDataFolder = os.path.join('/media/ak/OS', 'Data')
expFiles = os.path.join(laptopDataFolder, 'August11th2022Experiments')
destination = os.path.join(laptopDataFolder, 'AthensSoloTripFeb2023')
figuresDestination = '/home/ak/Documents/Papers/StylisedFactsPaper/figures/'

plt.style.use(os.path.join('/home/ak/.config/matplotlib', 'latexstyle3.mplstyle'))



#
#
class CrossCorrel:
    MICRO_VARIABLES = ['arrival_rates', 'gk_vol', 'median_traded_volume', 'micro_price_change']

    def __init__(self, path, symbols_list, idx, bar_choice, num_processes):
        self._idx = idx
        self._symbols = symbols_list
        self._bar = bar_choice
        self._symbol = self._symbols[self._idx]
        self._symbol_filepath = os.path.join(path, str(self._symbol))
        self._list_of_files = os.listdir(self._symbol_filepath)
        self._files_bar_symbol = [f for f in self._list_of_files if str(self._bar) in f]
        self._num_processes = num_processes

    def get_data_and_path(self):
        return self._files_bar_symbol, self._symbol_filepath

    def get_all_data_from_file(self, file_idx):
        file_to_get = os.path.join(self._symbol_filepath, self._files_bar_symbol[file_idx])
        return pd.read_pickle(file_to_get)

    def get_microvar_data(self, file_idx, var):
        pkl_dict = self.get_all_data_from_file(file_idx)
        return pkl_dict[str(var)]

    def get_all_microvar_data(self, file_idx):
        pkl_dict = self.get_all_data_from_file(file_idx)
        microvar_data = [pkl_dict[var] for var in self.MICRO_VARIABLES]
        bar_data = pkl_dict[str(self._bar)]
        tau_dict, alpha, mf_spect, n_f_dict, list_h_dict, list_h_intercept = [bar_data[var] for var in (
        'tau', 'alpha', 'mfSpect', 'n_F', 'list_H', 'list_H_intercept')]
        return (*microvar_data, tau_dict, alpha, mf_spect, n_f_dict, list_h_dict, list_h_intercept)

    @staticmethod
    def compute_n_rho(args):
        var_a, var_b, idx, win_sizes, pol_ord = args
        a = fu.toAggregated(var_a[idx])
        b = fu.toAggregated(var_b[idx])
        try:
            pydcca = fathon.DCCA(a, b)
            n, rho = pydcca.computeRho(win_sizes, polOrd=pol_ord)
        except ZeroDivisionError:
            return None, None
        return n, rho

    @staticmethod
    def compute_n_f_dcca(args):
        var_a, var_b, idx, win_sizes, pol_ord = args
        a = fu.toAggregated(var_a[idx])
        b = fu.toAggregated(var_b[idx])
        pydcca = fathon.DCCA(a, b)
        n = pydcca.computeFlucVec(win_sizes, polOrd=pol_ord)
        return n
    @staticmethod
    def compute_h_h_intercept_dcca(args):
        var_a, var_b, idx, win_sizes, pol_ord = args
        a = fu.toAggregated(var_a[idx])
        b = fu.toAggregated(var_b[idx])
        pydcca = fathon.DCCA(a, b)
        n, h = pydcca.computeHurstVec(win_sizes, polOrd=pol_ord)
        h_intercept = pydcca.computeHurstIntercept()
        return n, h, h_intercept

    @staticmethod
    def compute_n_rho_parallel(var_a, var_b, win_sizes, pol_ord):
        with mp.tool() as pool:
            results = [pool.apply_async(CrossCorrel.compute_n_rho, (var_a, var_b, idx, win_sizes, pol_ord)) for idx in
                       range(len(var_a))]
            n_rho_list = [result.get() for result in results]
        return n_rho_list
#
    @staticmethod
    def parallel_compute(var_a, var_b, win_sizes, pol_ord):
        num_procs = mp.cpu_count()
        pool = mp.Pool(num_procs)

        chunk_size = len(var_a) // num_procs
        idx_chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_procs)]
        idx_chunks[-1] = (idx_chunks[-1][0], len(var_a))

        n_rho_chunks = pool.starmap(
            CrossCorrel.compute_n_rho_parallel,
            [(var_a, var_b, range(chunk[0], chunk[1]), win_sizes, pol_ord) for chunk in idx_chunks]
        )

        n_rho_flat = [elem for chunk in n_rho_chunks for elem in chunk if elem is not None]
        n_flat, rho_flat = zip(*n_rho_flat)
        pool.close()

        return n_flat, rho_flat

    @staticmethod
    def parallel_compute_n_f_dcca(var_a, var_b, idx, win_sizes, pol_ord):
        a = fu.toAggregated(var_a[idx])
        b = fu.toAggregated(var_b[idx])
        pydcca = fathon.DCCA(a, b)

        with mp.Pool() as pool:
            results = pool.starmap(
                pydcca.computeFlucVec,
                [(win_size, pol_ord) for win_size in win_sizes]
            )
            n, F = zip(*results)

        return n, F

if __name__ == '__main__':

     tic = time.perf_counter()

     micro_variables = ['arrival_rates', 'gk_vol', 'median_traded_volume', 'micro_price_change']
     symbolsIdx = 6
     bar = 'volume'
     cc2 = CrossCorrel(t7DataFolder, symbols, symbolsIdx, str(bar), 5)

     files, filesPath = cc2.get_data_and_path()
     print(filesPath)
     print(len(os.listdir(filesPath)))
     range_to_use = range(0, 51)
     arrivals_list = {f: cc2.get_microvar_data(fileIdx=f, var='arrival_rates') for f in range_to_use}
     median_volumes_list = {f: cc2.get_microvar_data(fileIdx=f, var='median_traded_volume') for f in range_to_use}
     arrivalsDF = pd.DataFrame(arrivals_list).fillna(0)
     medianVolumesDF = pd.DataFrame(median_volumes_list).fillna(0)

     with Pool(5) as p:
        rho_dicts = p.map(cc2.parallel_compute_n_rho,
                          [(arrivalsDF, medianVolumesDF, idx, winSizes, polOrd) for idx in range(0, 20)])
        h_dcca_dicts = p.map(cc2.parallel_compute_H_H_dcca,
                             [(arrivalsDF, medianVolumesDF, idx, winSizes, polOrd) for idx in range(0, 40)])

    for idx, rho_dict in enumerate(rho_dicts):
        file_name = "_".join(('RhoDict_', str(symbols[symbolsIdx]), str(idx), bar, '.pkl'))
        pickle_out_filename = os.path.join(expInputFiles, file_name)
        pickle.dump(rho_dict, open(pickle_out_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    for idx, h_dcca_dict in enumerate(h_dcca_dicts):
        file_name = "_".join(('HurstDict_', str(symbols[symbolsIdx]), str(idx), bar, '.pkl'))
        pickle_out_filename = os.path.join(expInputFiles, file_name)
        pickle.dump(h_dcca_dict, open(pickle_out_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL))

