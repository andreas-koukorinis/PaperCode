import pandas as pd
from os.path import isfile, join
import numpy as np
import os
from os import listdir
import pickle
import datetime as dt
import json

# contains all the utilities for cleaning data#
data_dir = os.getenv('FINANCE_DATA')  # main directory referenced in all the code
fin_data_only = '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2/FinData/'  # external date only drive
raw_crypto_data = '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2/Data/crypto/LOB/BTCUSD.PERP.BMEX/' #btc perp data
data_drive='/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2/'
clean_files_data = "".join((data_drive,'/BTCUSD.PERP.BMEX/'))
low_memory=False
csv_files=[s for s in os.listdir(raw_crypto_data) if s.endswith('.csv')]
#cleaning first stack of crypto-data
for csv_files_idx, csv_date in enumerate(csv_files):
    # csv_files_idx=1
    csv_loc = "/".join((raw_crypto_data,csv_files[csv_files_idx]))
    raw_data = pd.read_csv(csv_loc,low_memory=False)
    clean_df = pd.DataFrame()
    clean_df[['happened_at', 'bestBidPrice', 'bestBidSize', 'bestAskPrice', 'bestAskSize']] = raw_data[
        ['happened_at', 'BIDPRC1', 'BIDSIZE1', 'ASKPRC1',
         'ASKSIZE1']]
    clean_df['timestamp'] = pd.to_datetime(clean_df['happened_at'])
    clean_df['time'] = clean_df['timestamp'].dt.time
    clean_df['date'] = clean_df['timestamp'].dt.date
    clean_df['milliSeconds'] = [(((x.hour * 60 + x.minute) * 60 + x.second) * 1000) for x in clean_df['time']]
    clean_df.drop('happened_at', axis=1, inplace=True)
    clean_df['duration'] = clean_df.milliSeconds.diff(1)
    clean_df['MicroPrice'] = (clean_df['bestAskPrice'] * clean_df['bestAskSize'] + clean_df['bestBidPrice'] * clean_df[
        'bestBidSize']) / (clean_df['bestAskSize'] + clean_df['bestBidSize'])
    clean_df['MicroPriceReturns'] = clean_df.MicroPrice.pct_change()
    clean_df = clean_df.dropna()
    df_name = "".join((clean_files_data,csv_date))
    clean_df.to_csv(df_name)



