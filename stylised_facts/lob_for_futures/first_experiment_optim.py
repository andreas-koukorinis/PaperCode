import time
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
from collections import defaultdict
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import jarque_bera
import itertools
import pickle
import datetime as dt
from datetime import timedelta
import lob_for_futures as lobFut
from lob_for_futures import *

elements = '/media/ak/Elements/OrderBookProcessedData'
experimentsLocation = '/media/ak/WorkDrive/Data/July2nd2022_Experiments'
bars = ['volume_bar', 'calendar_bar', 'usd_volume_bar', 'tick_bar']
sys.path.append('/home/ak/Documents/Research/PaperCode/stylised_facts')
import lob_for_futures as lobFut
from lob_for_futures import *

elements = '/media/ak/Elements/OrderBookProcessedData'
experimentsLocation = '/media/ak/WorkDrive/Data/July2nd2022_Experiments'
bars = ['volume_bar', 'calendar_bar', 'usd_volume_bar', 'tick_bar']

dataFolder = lobFut.dataFolder
quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]

symbols = sorted(os.listdir(elements))

def open_pickle_filepath(filepath: str):
    """
    Opens a pickle file at the specified filepath and returns its content.

    :param filepath: str, file path of the pickle file
    :return: content of the pickle file
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def returns(series: pd.Series) -> pd.Series:
    """
    Calculate returns for a given Pandas Series.

    :param series: pd.Series, input data series
    :return: pd.Series, calculated returns
    """
    return series.pct_change().dropna()

def standardise_returns(returns: pd.Series) -> pd.Series:
    """
    Standardise the returns by subtracting the mean and dividing by the standard deviation.

    :param returns: pd.Series, input returns series
    :return: pd.Series, standardised returns
    """
    return (returns - returns.mean()) / returns.std()

def process_returns(symbolIdx: int, filesIdx: int) -> defaultdict:
    """
    Calculate and save the standardised returns and Jarque-Bera statistics for a given symbol and file index.

    :param symbolIdx: int, index of the symbol in the symbols list
    :param filesIdx: int, index of the file in the files list
    :return: defaultdict, standardised returns dictionary
    """
    # Initialize the defaultdict for standardised returns
    standarised_returns = defaultdict(dict)

    # Define the file paths
    procsdSymbolFolder = os.path.join(elements, symbols[symbolIdx])
    files = sorted(os.listdir(procsdSymbolFolder))
    fileLocation = os.path.join(procsdSymbolFolder, files[filesIdx])

    # Load the data from the pickle files
    volume_bar_dict = open_pickle_filepath(fileLocation)[bars[0]]
    calendar_bar_dict = open_pickle_filepath(fileLocation)[bars[1]]
    usd_volume_bar_dict = open_pickle_filepath(fileLocation)[bars[2]]
    tick_bar_dict = open_pickle_filepath(fileLocation)[bars[3]]

    # Extract the DataFrames
    volume_bar_df = volume_bar_dict[list(volume_bar_dict.keys())[0]]
    calendar_bar_df = calendar_bar_dict[list(calendar_bar_dict.keys())[0]]
    usd_volume_df = usd_volume_bar_dict[list(usd_volume_bar_dict.keys())[0]]
    tick_bar_df = tick_bar_dict[list(usd_volume_bar_dict.keys())[0]]

    # Calculate the returns for each bar type
    vb_ret = returns(volume_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)

    tb_ret = returns(tick_bar_df.import time
sys.path.append('/home/ak/Documents/Research/PaperCode/stylised_facts')

symbols = sorted(os.listdir(elements))

def open_pickle_filepath(filepath: str):
    """
    Opens a pickle file at the specified filepath and returns its content.

    :param filepath: str, file path of the pickle file
    :return: content of the pickle file
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def returns(series: pd.Series) -> pd.Series:
    """
    Calculate returns for a given Pandas Series.

    :param series: pd.Series, input data series
    :return: pd.Series, calculated returns
    """
    return series.pct_change().dropna()

def standardise_returns(returns: pd.Series) -> pd.Series:
    """
    Standardise the returns by subtracting the mean and dividing by the standard deviation.

    :param returns: pd.Series, input returns series
    :return: pd.Series, standardised returns
    """
    return (returns - returns.mean()) / returns.std()

def process_returns(symbolIdx: int, filesIdx: int) -> defaultdict:
    """
    Calculate and save the standardised returns and Jarque-Bera statistics for a given symbol and file index.

    :param symbolIdx: int, index of the symbol in the symbols list
    :param filesIdx: int, index of the file in the files list
    :return: defaultdict, standardised returns dictionary
    """
    # Initialize the defaultdict for standardised returns
    standarised_returns = defaultdict(dict)

    # Define the file paths
    procsdSymbolFolder = os.path.join(elements, symbols[symbolIdx])
    files = sorted(os.listdir(procsdSymbolFolder))
    fileLocation = os.path.join(procsdSymbolFolder, files[filesIdx])

    # Load the data from the pickle files
    volume_bar_dict = open_pickle_filepath(fileLocation)[bars[0]]
    calendar_bar_dict = open_pickle_filepath(fileLocation)[bars[1]]
    usd_volume_bar_dict = open_pickle_filepath(fileLocation)[bars[2]]
    tick_bar_dict = open_pickle_filepath(fileLocation)[bars[3]]

    # Extract the DataFrames
    volume_bar_df = volume_bar_dict[list(volume_bar_dict.keys())[0]]
    calendar_bar_df = calendar_bar_dict[list(calendar_bar_dict.keys())[0]]
    usd_volume_df = usd_volume_bar_dict[list(usd_volume_bar_dict.keys())[0]]
    tick_bar_df = tick_bar_dict[list(usd_volume_bar_dict.keys())[0]]

    # Calculate the returns for each bar type
    vb_ret = returns(volume_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)
    tb_ret = returns(tick_bar_df.micro_price_close).replace([np.inf, -np.inf], 0)
    usdvb_ret = returns(usd_volume_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)
    cb_ret = returns(calendar_bar_df.micro_price_close).dropna().replace([np.inf, -np.inf], 0)

    # Standardise the returns for each bar type
    tick_standard = standardise_returns(tb_ret)
    volume_standard = standardise_returns(vb_ret)
    dollar_standard = standardise_returns(usdvb_ret)
    cb_standard = standardise_returns(cb_ret)

    # Save the standardised returns to the defaultdict
    standarised_returns['tick'][filesIdx] = tick_standard
    standarised_returns['calendar'][filesIdx] = cb_standard
    standarised_returns['volume'][filesIdx] = volume_standard
    standarised_returns['dollar'][filesIdx] = dollar_standard

    # Save the standardised returns to a pickle file
    pickle_out_returns = os.path.join(experimentsLocation,
                                      "".join((str(symbols[symbolIdx]), "_" + str(filesIdx) + "_standard_returns.pkl")))
    pickle.dump(standarised_returns, open(pickle_out_returns, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate the Jarque-Bera statistics for each bar type
    jb_value_tick, _ = jarque_bera(tb_ret)
    jb_value_vol, _ = jarque_bera(vb_ret)
    jb_value_dollar, _ = jarque_bera(usdvb_ret)
    jb_value_calendar, _ = jarque_bera(cb_ret)

    # Save the Jarque-Bera statistics to a DataFrame
    jb_test_df = pd.DataFrame(data={'jarque_bera_results': [jb_value_tick,
                                                            jb_value_vol,
                                                            jb_value_dollar,
                                                            jb_value_calendar]
                                    },
                              index=['tick', 'vol', 'dollar', 'calendar'])

    # Save the Jarque-Bera statistics to a pickle file
    pickle_out_jb = os.path.join(experimentsLocation,
                                 "".join((str(symbols[symbolIdx]), "_" + str(filesIdx) + "_jb_stats.pkl")))
    pickle.dump(jb_test_df, open(pickle_out_jb, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    return standarised_returns

if __name__ == '__main__':
    a_args = [3] # symbol
    second_arg = [f for f in range(0, 100, 1)]  # range of files
    freeze_support()
    # Calculate and save standardised returns and Jarque-Bera statistics
    with Pool() as pool:
        results = pool.starmap(process_returns, list(itertools.product(a_args, second_arg)))
    second_arg = [f for f in range(0, 100, 1)] # range of files

