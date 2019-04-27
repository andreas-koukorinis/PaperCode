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
clean_files_data = "".join((data_drive,'/BTCUSD.PERP.BMEX'))

csv_files=[s for s in os.listdir(raw_crypto_data) if s.endswith('.csv')]
csv_files_idx=1
csv_loc = "/".join((raw_crypto_data,csv_files[csv_files_idx]))
raw_data=pd.read_csv(csv_loc)
clean_df=pd.DataFrame()
clean_df[['happened_at','BID','BIDSIZE','ASK','ASKSIZE']]=raw_data[['BIDPRC1', 'BIDSIZE1', 'ASKPRC1',
       'ASKSIZE1']]
clean_df['timestamp']=pd.to_datetime(clean_df['happened_at'])
clean_df['time']=clean_df['timestamp'].dt.time
clean_df['date']=clean_df['timestamp'].dt.date
clean_df['milliSeconds']=[int(((x.hour*60 + x.minute)*60 + x.second)*1000) for x in clean_df['time']]
clean_df.drop('happened_at', axis=1, inplace=True)
print clean_df.head(10)
