import pandas as pd
import fathon
from fathon import fathonUtils as fu
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import os
from collections import defaultdict
from multiprocessing import Pool, freeze_support
import time
import itertools

sys.path.append('/home/ak/Documents/PaperCode/stylised_facts')
sys.path.append('/home/ak/Research/PaperCode/stylised_facts')
import numpy as np
from mdfda import mdfda_experiments_utils as mdf
import pickle

scaler = MinMaxScaler()
standard_scaler = StandardScaler()

sys.path.insert(0, '/directory/tothe/handshakefile/')


def read_pkl_idx(file_loc, file_idx):
    files = os.listdir(file_loc)
    file_idx_loc = os.path.join(file_loc, files[file_idx])
    dict_idx = pd.read_pickle(file_idx_loc)
    return dict_idx


def mfdfa_input_rx(bars_dict, bar_type_):
    df_output = bars_dict[bar_type_]
    return df_output


def mfdfa_output(lob_df_input, bar_type_):
    """
    function that takes input:
    indexed dictionary of bars, chosen bar type
    lob_df_input: this is the precise experiment dataframe
    returns mfdfa output
    NB: needs to get edited to take column type as well
    NB 2: needs a params file
    """
    raw_data = lob_df_input.pct_change_micro_price  # this is the chosen data to use
    # follows Fathon Example from here onwards
    data_input = mdf.to_agg(raw_data)
    try:
        winSizes = fu.linRangeByStep(10, 2000)  # this needs to be more dynamic
        qs = np.arange(-3, 4, 0.1)
        revSeg = True
        polOrd = 1
        testClass = mdf.mfdfaquantities(data_input, winSizes, qs, revSeg, polOrd)
        n, F = testClass.n_F_output()
        list_H, list_H_intercept = testClass.H_and_H_intcpt_output()
        tau = testClass.compute_mass_exponents()
        alpha, mfSpect = testClass.compute_multi_fractal_spectrum()

        mfdfa_output_dict_ = defaultdict(dict)
        mfdfa_output_dict_['micro_price_change'] = raw_data
        mfdfa_output_dict_['arrival_rates'] = lob_df_input.arrival_rates
        mfdfa_output_dict_['gk_vol'] = lob_df_input.GK_vol
        mfdfa_output_dict_['median_traded_volume'] = lob_df_input.median_traded_volume
        mfdfa_output_dict_[bar_type_]['n_F'] = dict(zip(n, F))
        mfdfa_output_dict_[bar_type_]['list_H'] = list_H
        mfdfa_output_dict_[bar_type_]['list_H_intercept'] = list_H_intercept
        mfdfa_output_dict_[bar_type_]['tau'] = tau
        mfdfa_output_dict_[bar_type_]['alpha'] = alpha
        mfdfa_output_dict_[bar_type_]['mfSpect'] = mfSpect
    except ValueError:
        pass

    return mfdfa_output_dict_


def all_in_calculations(symbol_input_files_loc, symbol_file_idx_, bar_type_):
    print('doing index', symbol_file_idx_)
    df = (read_pkl_idx(symbol_input_files_loc, symbol_file_idx_))
    mfdfa_dict_output = mfdfa_output(df, str(bar_type))
    test_output_loc = os.path.join(expOneLocation,
                                   str(symbol_) + '_' + str(symbol_file_idx_) + str(bar_type_) + "_mfdfa.pkl")
    pickle.dump(mfdfa_dict_output, open(test_output_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('saved')
    return mfdfa_dict_output


experimentsLocation = '/media/ak/Elements/August11th2022Experiments/'
expOneLocation = os.path.join(experimentsLocation, 'ExperimentOne')

if __name__ == '__main__':
    symbol_ = 'US1'
    # symbol_input_files_loc_ = os.path.join(experimentsLocation, 'ExperimentInputFiles')

    st = time.time()
    symbol = symbol_
    bar_type = 'volume'
    symbol_input_files_loc = os.path.join(experimentsLocation, 'ExperimentInputFiles', symbol)
    symbol_files = [f for f in os.listdir(symbol_input_files_loc) if str(bar_type) in f]
    pool = Pool(4)
    start_time = time.perf_counter()
    print()
    a_args = [f for f in range(0, len(symbol_files), 1)]  # length of files
    print(a_args)
    b_args = ['volume']  # range of bars
    # print(all_in_calculations(symbol_input_files_loc, 10, bar_type))
    freeze_support()
    # takes bar and file and produces the output
    with Pool() as pool:
        L = pool.starmap(all_in_calculations, [(symbol_input_files_loc, symbol_file_idx_, bar_type_)
                                               for symbol_input_files_loc in [symbol_input_files_loc]
                                               for symbol_file_idx_ in a_args for bar_type_ in b_args])
    et = time.time()
    elapsed = et - st
    print('Elapsed time: ', elapsed)
