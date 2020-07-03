import pandas as pd
from os.path import isfile, join
import numpy as np
import os
from dateutil.parser import parse
import datetime as dt
from numba import jit

from tqdm import tqdm, tqdm_notebook

# contains all the utilities for cleaning data#
# data_only_drive = '/media/ak/DataOnly/'  # external date only drive
# clean_files_data = "".join((data_only_drive,'/BTCUSD.PERP.BMEX/'))
# raw_crypto_data_lob = '/media/ak/DataOnly/crypto/LOB/BTCUSD.PERP.BMEX/' #btc perp data
# clean_crypto_files = os.listdir(clean_files_data)
# csv_files = [s for s in os.listdir(raw_crypto_data_lob) if s.endswith('.csv')]
low_memory = False


# useful functions###


def micro_price(df):

    df['MicroPrice']=(df['bestAskPrice']*df['bestAskSize']+df['bestBidPrice']*df['bestBidSize'])/(df['bestAskSize']+df['bestBidSize'])
    return df


def returns(s):

    # compute log returns
    arr = np.diff(np.log(s))

    return (pd.Series(arr, index=s.index[1:]))

@jit(nopython=True)
def mad_outlier(y, thresh=3.):
    '''
    compute outliers based on mad
    # args
        y: assumed to be array with shape (N,1)
        thresh: float()
    # returns
        array index of outliers
    '''
    median = np.median(y)
    diff = np.sum((y - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
# # cleaning first stack of crypto-data
# mylist = []
# fields = ['happened_at', 'BIDPRC1', 'BIDSIZE1', 'ASKPRC1', 'ASKSIZE1']
#
# # for csv_files_idx, csv_date in enumerate(csv_files):
# #     print csv_date
# csv_files_idx=1
# csv_loc = "/".join((raw_crypto_data_lob,csv_files[csv_files_idx]))
# for chunk in pd.read_csv(csv_loc, chunksize=2000, skipinitialspace=True, usecols=fields, error_bad_lines=False, engine='python'):
#     mylist.append(chunk)
#     big_data_df = pd.concat(mylist, axis=0)
# del mylist
# print df_name
# df_name = "".join((clean_files_data,csv_files_idx,".csv"))
# big_data_df.to_csv(df_name)
# # raw_data = pd.read_csv(csv_loc,low_memory=False, error_bad_lines=False)
#     # clean_df = pd.DataFrame()
#     # clean_df[['happened_at', 'bestBidPrice', 'bestBidSize', 'bestAskPrice', 'bestAskSize']] = raw_data[
#     #     ['happened_at', 'BIDPRC1', 'BIDSIZE1', 'ASKPRC1',
#     #      'ASKSIZE1']]
#     # clean_df['timestamp'] = pd.to_datetime(clean_df['happened_at'])
#     # clean_df['time'] = clean_df['timestamp'].dt.time
#     # clean_df['date'] = clean_df['timestamp'].dt.date
#     # clean_df['milliSeconds'] = [(((x.hour * 60 + x.minute) * 60 + x.second) * 1000) for x in clean_df['time']]
#     # clean_df.drop('happened_at', axis=1, inplace=True)
#     # clean_df['duration'] = clean_df.milliSeconds.diff(1)
#     # clean_df['MicroPriceReturns'] = clean_df.MicroPrice.pct_change()
#     # clean_df = clean_df.dropna()
#     # print clean_df.head(5)
#     # final_df = micro_price(clean_df)
#     # final_df['LogMicroPriceReturns']=returns(final_df.MicroPrice)
#     # final_df= final_df.dropna()
#     # df_name = "".join((clean_files_data,csv_date))
#     # final_df.to_csv(df_name)
#
#
#
#
