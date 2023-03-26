import pandas as pd
from fathon import fathonUtils as fu
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import os
from collections import defaultdict
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import time
import argparse
import itertools
from mdfda import mdfda_experiments_utils as mdf
import pickle

scaler = MinMaxScaler()
standard_scaler = StandardScaler()

sys.path.append('/home/ak/Documents/PaperCode/stylised_facts')
sys.path.append('/home/ak/Research/PaperCode/stylised_facts')
sys.path.insert(0, '/directory/tothe/handshakefile/')

# locations
experimentsLocation = '/media/ak/T71/August11th2022Experiments'
expOneLocation = os.path.join(experimentsLocation, 'ExperimentOne')


# functions

def read_pkl_idx(file_loc, file_idx):
    """
    Reads a pickle file based on the given file location and index.

    :param file_loc: str, file location containing the pickle files
    :param file_idx: int, index of the file to read
    :return: dict, content of the pickle file
    """
    files = os.listdir(file_loc)
    file_idx_loc = os.path.join(file_loc, files[file_idx])
    dict_idx = pd.read_pickle(file_idx_loc)
    return dict_idx


def mfdfa_input_rx(bars_dict, bar_type_):
    """
    Returns a dataframe from the bars dictionary based on the given bar type.

    :param bars_dict: dict, dictionary containing dataframes for different bar types
    :param bar_type_: str, bar type to extract from the dictionary
    :return: DataFrame, dataframe corresponding to the given bar type
    """
    df_output = bars_dict[bar_type_]
    return df_output


def mfdfa_output(lob_df_input, bar_type_):
    """
    Calculates the MFDFA quantities for the input DataFrame.
    Function that takes input:
    indexed dictionary of bars, chosen bar type
    lob_df_input: this is the precise experiment dataframe
    returns mfdfa output
    NB: needs to get edited to take column type as well
    NB 2: needs a params file
    """
    if lob_df_input is None:
        return None
    else:

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


def process_symbol_files(symbol_input_files_loc, symbol_file_idx_, bar_type_, symbol):
    """
       Processes a single symbol file with the given index and bar type.

       :param symbol_input_files_loc: str, file location containing the symbol files
       :param symbol_file_idx_: int, index of the file to process
       :param bar_type_: str, bar type to be used
       :param symbol: str, symbol for the current file
       :return: defaultdict, dictionary containing the MFDFA output
       """
    print('doing index', symbol_file_idx_)
    df = (read_pkl_idx(symbol_input_files_loc, symbol_file_idx_))
    mfdfa_dict_output = mfdfa_output(df, str(bar_type_))
    test_output_loc = os.path.join(expOneLocation,
                                   str(symbol) + '_' + str(symbol_file_idx_) + str(bar_type_) + "_mfdfa.pkl")
    pickle.dump(mfdfa_dict_output, open(test_output_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('saved')
    return mfdfa_dict_output


def main_list(symbols, bar_type):
    pool = Pool()  # Use the default number of processes
    for symbol in symbols:
        symbol_input_files_loc = os.path.join(experimentsLocation, 'ExperimentInputFiles', symbol)
        symbol_files = [f for f in os.listdir(symbol_input_files_loc) if bar_type in f]
        a_args = [f for f in range(0, len(symbol_files), 1)]  # length of files
        b_args = [bar_type]  # range of bars
        with pool:
            L = pool.starmap(all_in_calculations, [(symbol_input_files_loc, symbol_file_idx_, bar_type)
                                                   for symbol_input_files_loc in [symbol_input_files_loc]
                                                   for symbol_file_idx_ in a_args for bar_type_ in b_args])
    pool.close()
    pool.join()


def all_in_calculations(symbol_input_files_loc, symbol_file_idx, bar_type, symbol):
    """
    Performs calculations for a specific symbol file and saves the result as a pickle file.

    :param symbol_input_files_loc: str, path to the location of the symbol input files
    :param symbol_file_idx: int, index of the symbol file to process
    :param bar_type: str, type of the bar (e.g., 'tick')
    :param symbol: str, symbol to be processed (e.g., 'FB1')
    :return: dict, the output dictionary containing the results of the MFDFA calculations
    """
    print('doing index', symbol_file_idx)
    df = (read_pkl_idx(symbol_input_files_loc, symbol_file_idx))
    mfdfa_dict_output = mfdfa_output(df, bar_type)
    test_output_loc = os.path.join(expOneLocation, f"{symbol}_{symbol_file_idx}{bar_type}_mfdfa.pkl")
    pickle.dump(mfdfa_dict_output, open(test_output_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('saved')
    return mfdfa_dict_output


def main(symbols, bar_types, experimentsLocation, expOneLocation):
    """
    Main function to process multiple symbols and bar types using parallel processing.

    :param symbols: list of str, a list of symbols to be processed
    :param bar_types: list of str, a list of bar types to be processed
    :param experimentsLocation: str, path to the location of the experiments
    :param expOneLocation: str, path to the location of the ExperimentOne folder
    """
    for symbol in symbols:
        for bar_type in bar_types:
            symbol_input_files_loc = os.path.join(experimentsLocation, 'ExperimentInputFiles', symbol)
            symbol_files = [f for f in os.listdir(symbol_input_files_loc) if bar_type in f]
            symbol_files_count = len(symbol_files)

            with Pool() as pool:
                pool.starmap(
                    all_in_calculations,
                    [(symbol_input_files_loc, symbol_file_idx, bar_type, symbol) for symbol_file_idx in
                     range(symbol_files_count)]
                )


if __name__ == '__main__':
    symbols_list = ['FV1']  # Add more symbols to the list if needed
    bar_types_list = ['dollar']  # Add more bar types to the list if needed
    experimentsLocation = '/media/ak/T71/August11th2022Experiments'
    expOneLocation = os.path.join(experimentsLocation, 'ExperimentOne')
    main(symbols_list, bar_types_list, experimentsLocation, expOneLocation)
