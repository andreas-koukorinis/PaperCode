import pandas as pd
from fathon import fathonUtils as fu
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from mdfda import mdfda_experiments_utils as mdf
import pickle

scaler = MinMaxScaler()
standard_scaler = StandardScaler()


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


def mfdfa_output(lob_df_input: pd.DataFrame, _bar_type_: str) -> dict:
    """
    Processes a given pandas DataFrame to calculate MFDFA.
    If the input DataFrame is None, it simply returns None. Otherwise, it performs the MFDFA calculation.
    :param lob_df_input: pandas.DataFrame, input dataframe
    :param _bar_type_: str, bar type
    :return: dict, dictionary containing the MFDFA output
    """
    if lob_df_input is None:
        return None
    else:
        raw_data = lob_df_input.pct_changes.fillna(0)  # this is the chosen data to use
        # follows Fathon Example from here onwards
        data_input = mdf.to_agg(raw_data)
    try:
        # Create the winSizes list based on the size of the data
        # Here I'm making the upper limit of the size of the data less one element
        max_window_size = len(raw_data) - 1
        winSizes = fu.linRangeByStep(5, max_window_size)

        qs = np.arange(-3, 4, 0.1)
        revSeg = True
        polOrd = 1
        testClass = mdf.mfdfaquantities(data_input, winSizes, qs, revSeg, polOrd)
        n, F = testClass.n_F_output()
        list_H, list_H_intercept = testClass.H_and_H_intcpt_output()
        alpha, mfSpect = testClass.compute_multi_fractal_spectrum()

        mfdfa_output_dict_ = defaultdict(dict)
        mfdfa_output_dict_[_bar_type_]['micro_price_change'] = raw_data
        mfdfa_output_dict_[_bar_type_]['arrival_rate_median'] = lob_df_input.arrival_rate_median
        mfdfa_output_dict_[_bar_type_]['simple_vol'] = lob_df_input.simple_vol
        mfdfa_output_dict_[_bar_type_]['median_traded_volume'] = lob_df_input.TradedVolume_median
        mfdfa_output_dict_[_bar_type_]['n_F'] = dict(zip(n, F))
        mfdfa_output_dict_[_bar_type_]['list_H'] = list_H
        mfdfa_output_dict_[_bar_type_]['list_H_intercept'] = list_H_intercept
        mfdfa_output_dict_[_bar_type_]['tau'] = testClass.compute_mass_exponents()
        mfdfa_output_dict_[_bar_type_]['alpha'] = alpha
        mfdfa_output_dict_[_bar_type_]['mfSpect'] = mfSpect
    except (ValueError, AttributeError, TypeError):
        pass

    return mfdfa_output_dict_


def process_symbol_files(symbol_input_files_locn, symbol_file_index_, bar_type_, symbol):
    """
    Processes a single symbol file with the given index and bar type.
    :param symbol_input_files_locn: str, file location containing the symbol files
    :param symbol_file_index_: int, index of the file to process
    :param bar_type_: str, bar type to be used
    :param symbol: str, symbol for the current file
    :return: defaultdict, dictionary containing the MFDFA output
    """
    print('doing index', symbol_file_index_)

    df = (read_pkl_idx(symbol_input_files_locn, symbol_file_index_))

    mfdfa_dict_output = mfdfa_output(df, str(bar_type_))

    print('will be saved')
    print(mfdfa_dict_output)

    return mfdfa_dict_output


def all_in_calculations(input_files_loctns: str, symbol_file_idx: int, bar_type: str, symbol: str) -> dict:
    """
    Performs calculations for a specific symbol file and saves the result as a pickle file.
    :param input_files_loctns: str, path to the location of the symbol input files
    :param symbol_file_idx: int, index of the symbol file to process
    :param bar_type: str, type of the bar (e.g., 'tick')
    :param symbol: str, symbol to be processed (e.g., 'FB1')
    :return: dict, the output dictionary containing the results of the MFDFA calculations
    """
    print('Processing index', symbol_file_idx)
    # symbol_input_files_location = os.path.join(input_files_loctns, symbol, bar_type)

    df = read_pkl_idx(input_files_loctns, symbol_file_idx)
    mfdfa_dict_output = mfdfa_output(df, bar_type)

    output_file_name = str(os.listdir(input_files_loctns)[symbol_file_idx].split(".")[0]) + "_mfdfa.pkl"
    test_output_loc = os.path.join(mainMFDFAPath, symbol, bar_type, output_file_name)

    pickle.dump(mfdfa_dict_output, open(test_output_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved in', test_output_loc)

    return mfdfa_dict_output


def main(inputLoc_: str, symbols: list, bar_types: list, max_workers=10):
    """
    Main function to process multiple symbols and bar types using parallel processing.
    :param symbols: list of str, a list of symbols to be processed
    :param bar_types: list of str, a list of bar types to be processed
    :param inputLoc_: str, path to the location of the input files
    """
    with ThreadPool(max_workers) as pool:
        for symbol in symbols:
            for bar_type in bar_types:
                symbol_int_files_loc = os.path.join(inputLoc_, symbol, bar_type)
                if not os.path.isdir(symbol_int_files_loc):
                    print(f'Skipping {symbol_int_files_loc}, directory not found.')
                    continue

                symbol_files_count = len(os.listdir(symbol_int_files_loc))
                if symbol_files_count == 0:
                    print('Skipping, no files found.')
                    continue

                print(f'Processing {symbol_files_count} files in {symbol_int_files_loc}')

                pool.starmap(all_in_calculations,
                             [(symbol_int_files_loc, i, bar_type, symbol) for i in range(symbol_files_count)])


if __name__ == '__main__':
    mainDataPath = '/media/ak/Data/InterestRateFuturesData'
    mainMFDFAPath = os.path.join(mainDataPath, 'MFDFA')
    eventClockPaths = '/media/ak/Data/InterestRateFuturesData/EventClocksFiles'
    # symbols_list = ['DU1', 'FB1', 'FV1', 'JB1', 'KE1', 'OE1', 'RX1', 'US1', 'US2y', 'XM1',
    #                 'YM1']  # Add more symbols to the list if needed
    bar_types_list = ['dollar']  # Add more bar types to the list if needed
    inputLocation = eventClockPaths
    symbols_list = ['RX1']
    main(inputLocation, symbols_list, bar_types_list)
