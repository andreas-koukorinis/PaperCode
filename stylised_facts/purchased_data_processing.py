import multiprocessing.pool

import pandas as pd
import os
import numpy as np
from functools import partial
import dask.dataframe as dd
import dask.multiprocessing
from multiprocessing import Pool
import time

dask.config.set(scheduler='processes')
import time

import dask.array as da
import dask.bag as db

if __name__ == '__main__':
    symbol = 'ES'  # pick a symbol from the list- start with these ['CL', 'ES', 'FLG', 'FTSE', 'JNI', 'STXE']
    # the first part takes the files that are stored in- basically just picks the csvs
    symbol_path = os.path.join('/media/ak/Data1/TradesPurchased', symbol)  # this is the path
    targetLocation = '/media/ak/Data1/ProcessedFuturesData'  # this is where the processed files will be
    symbol_target_location = os.path.join(targetLocation, symbol)  # here is the symbol location for the processed files
    symbol_csv = sorted([f for f in os.listdir(symbol_path) if str('.gz') not in f])  # these are the csvs

    symbol_file_idx = 1  # this we will need to loop over or figure out the multi-processing
    file_loc = os.path.join(symbol_path, symbol_csv[symbol_file_idx])  # location of said file above

    ## the dask bit starts here

    start = time.time()
    dask_df = dd.read_csv(file_loc, assume_missing=True)  # reading the dask dataframe
    end = time.time()
    print("Read csv with dask: ", (end - start), "sec")
    symbol_unique_dates = np.unique(list(dask_df['Date'].values.compute()))


    def read_process_store(uniqueDateIdx, dask_df, symbol_unique_dates):
        start1 = time.time()

        # for this file these are all the unique dates that are in there
        # get the dates out
        # pick one date
        dt = pd.to_datetime(symbol_unique_dates[uniqueDateIdx])
        # converting into a format that can be used
        dt_save_format = dt.strftime('%Y%m%d') + '.csv'
        # this will be the format to use when saving the file in the target location
        df_unique_date_symbol = dask_df[dask_df.Date == symbol_unique_dates[uniqueDateIdx]].compute()
        df_trades_symbol = df_unique_date_symbol[df_unique_date_symbol.Type == "Trade"].drop(['Bid Price', 'Bid Size','Ask Price', 'Ask Size'], 1).groupby('Time', as_index=False, sort=True)['Date','Price','Volume'].mean()
        # trades and
        df_quotes_symbol = df_unique_date_symbol[df_unique_date_symbol.Type == "Quote"].drop(['Price', 'Volume','Type'], 1).groupby('Time', as_index=False, sort=True)['Date','Bid Price','Bid Size','Ask Price','Ask Size'].mean()
        # quotes
        quotes_file_name = os.path.join(symbol_target_location, 'Quote_'+ dt_save_format)
        trades_file_name = os.path.join(symbol_target_location, 'Trades_' + dt_save_format)
        print(quotes_file_name)

        df_quotes_symbol.to_csv(quotes_file_name, index=False)
        print('------------')
        df_trades_symbol.to_csv(trades_file_name, index=False)
        end1 = time.time()
        print("Finish processing data: ", (end1 - start1), "sec")

    unique_date_idx = 12
    unique_dates_inputs = list(range(len(symbol_unique_dates)))
    for unique_date_idx in unique_dates_inputs:

        read_process_store(dask_df=dask_df, symbol_unique_dates= symbol_unique_dates,
                           uniqueDateIdx = unique_date_idx)
        #
    #
    # def parallel_runs(data_list):
    #     pool = multiprocessing.Pool(6)
    #     prod_x = partial(read_process_store,dask_df=dask_df, symbol_unique_dates= symbol_unique_dates )
    #     try:
    #         results = pool.map(prod_x, data_list)
    #         print(results)
    #     except KeyError:
    #         pass
    #
    #
    # parallel_runs(unique_dates_inputs)
    #
