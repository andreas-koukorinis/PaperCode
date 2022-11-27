import multiprocessing.pool
from datetime import datetime
import pandas as pd
import pickle
import os
import numpy as np
from functools import partial
import dask.dataframe as dd
import dask.multiprocessing
from multiprocessing import Pool, freeze_support
import time
import itertools

LOB = pd.DataFrame()
import datetime
import os
import dask.dataframe as dd
from pathlib import Path


def write_to_csv_for_symbol(file_loc, write_data_folder):
    ## Get symbol name:
    symbol = file_loc.split('/')[-1].split('.')[0]

    ## Need to use dask delayed if this doesn't work
    ddf = dd.read_csv(file_loc, assume_missing=True)
    quotes_ddf = ddf[ddf.Type == 'Quote']
    trades_ddf = ddf[ddf.Type == 'Trade']

    write_to_csv_for_dates(quotes_ddf, suffix='quotes', cols_to_drop=['Price', 'Volume', 'Type'],
                           output_data_folder=write_data_folder)
    write_to_csv_for_dates(trades_ddf, suffix='trades',
                           cols_to_drop=['Type', 'Bid Price', 'Bid Size', 'Ask Price', 'Ask Size'],
                           output_data_folder=write_data_folder)


def write_to_csv_for_dates(ddf, suffix, cols_to_drop, output_data_folder):
    ddf_dates = sorted(ddf['Date'].unique().compute())

    for d in ddf_dates:
        date = datetime.datetime.strptime(d, '%m/%d/%Y').date()
        quotes_for_date_ddf = ddf[ddf.Date == d].drop(cols_to_drop, axis=1).compute()
        # # print(quotes_for_date_ddf.head(10))
        # print(type(quotes_for_date_ddf))
        write_file_dir = os.path.join(output_data_folder, f"{symbol}_{suffix}")
        Path(write_file_dir).mkdir(parents=True, exist_ok=True)
        write_file_loc = os.path.join(write_file_dir, f"{datetime.datetime.strftime(date, '%Y%m%d')}.csv")
        quotes_for_date_ddf.to_csv(write_file_loc)


if __name__ == '__main__':
    data1 = '/media/ak/Data1/'
    trades_quotes_folder = os.path.join(data1, 'TradesPurchased')
    processed_sampled_data = os.path.join(data1, 'ProcessedSampledData/')
    symbols = os.listdir(trades_quotes_folder)
    symbol = 'STXE'
    symbol_trades_quotes_folder = os.path.join(trades_quotes_folder, symbol, 'CSV')
    symbol_files = sorted(os.listdir(symbol_trades_quotes_folder))
    write_data_folder = os.path.join(trades_quotes_folder, 'tmp')
    def all_in_calculations(fileIdx):
        idx_symbol_file = os.path.join(symbol_trades_quotes_folder, fileIdx)
        print(idx_symbol_file)
        write_to_csv_for_symbol(idx_symbol_file, write_data_folder)

    # a_args = [3]  # symbol
    # second_arg = symbol_files  # range of files

    st = time.time()

    with Pool() as pool:
        L = pool.map(all_in_calculations, list(symbol_files))
    et = time.time()
    elapsed = et - st
    print('Elapsed time: ', elapsed)
    #
    # for _, fileIdx in enumerate(symbol_files):
    #     idx_symbol_file = os.path.join(symbol_trades_quotes_folder, fileIdx)
    #     print(idx_symbol_file)
    #     write_to_csv_for_symbol(idx_symbol_file, write_data_folder)
