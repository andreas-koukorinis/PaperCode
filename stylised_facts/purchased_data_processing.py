import pandas as pd
import os
import numpy as np
import dask.dataframe as dd
import dask.multiprocessing
dask.config.set(scheduler='processes')
import time

import dask.array as da
import dask.bag as db
if __name__ == '__main__':
    CL = '/media/ak/Data1/TradesPurchased/CL'
    CL_csv = sorted([f for f in os.listdir(CL) if str('.gz') not in f])
    CL_file_idx = 1
    file_loc = os.path.join(CL, CL_csv[CL_file_idx])

    start = time.time()
    dask_df = dd.read_csv(file_loc, assume_missing=True)
    end = time.time()
    print("Read csv with dask: ", (end - start), "sec")

    print(dask_df['Bid Price'].mean().compute())