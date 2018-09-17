import pandas as pd
import sys
import datetime
from datetime import date
import os, fnmatch
import numpy as np
_path=('/home/ak/Documents/Data/GPDATA/')
import shutil
### make a list of all the files, this will be used to parse through ###
### ignore files that are there before ###
_files=fnmatch.filter(os.listdir(_path), 'garethpet*.csv')
sys.path.append(_path)

no_files=len(_files)

###define functions which will be used###

def _raw_data(file_name):
    raw_data =pd.read_csv(file_name, low_memory=False)
    return raw_data

def _columns(data_frame):
    _columns_ =data_frame.columns.values
    return _columns_

def _trade_data(data_frame):
    trade_data = data_frame[data_frame['Type'] == 'Trade']
    return trade_data


def _basic_columns(data_frame):
    basic_columns=data_frame[['#RIC','Date[G]','Time[G]','Price','Volume']]
    return basic_columns

def _date_convert(_date):
    return datetime.date.strftime(datetime.datetime.strptime(_date, '%d-%b-%Y'), '%Y%m%d')

def time_in_millis(time_stamp):
    conv_time =datetime.datetime.strptime(time_stamp, '%H:%M:%S.%f')
    in_millis=(((conv_time.hour.real*60 + conv_time.minute.real) *60)
               +(conv_time.second.real))*1000 +float(conv_time.microsecond)/10**3.
    return in_millis

def listdirs(folder): #return only directories from a master folder
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def copy_directory(dir_src, dir_dst ): #copy all files from one directory with one extension to another
    for filename in os.listdir(dir_src):
        if filename.endswith('.csv'):
            shutil.copy(dir_src + filename, dir_dst)

def ensure_dir(file_path): #ensure a dictory exists otherwise create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

####


ticker_data={}
date_data={}
data_trades={}
raw_data=  pd.read_csv('/home/ak/Documents/Data/GPDATA/garethpeters@unsw.edu.au-Jan2012_HMMProj-N145855118.csv.tar.gz', compression='gzip')
raw_data.columns = ['#RIC' if x=='garethpeters@unsw.edu.au-Jan2012_HMMProj-N145855118.csv' else x for x in raw_data.columns]

for i in range(no_files):
     print i
     #raw_data= comp_data #_raw_data(_path + _files[i])
     data_trades[i] =_trade_data(raw_data)
     data_trades[i]= _basic_columns(data_trades[i])
     data_trades[i].rename(columns={'Price': 'TradedPrice', 'Time[G]': 'TradedTime'}, inplace=True)
     data_trades[i]['ReturnTradedPrice'] =data_trades[i]['TradedPrice'].pct_change()
     data_trades[i]['Duration']= data_trades[i]['TradedTime'].apply(time_in_millis).diff().astype(float)
     data_trades[i]['logReturnTradedPrice']  = np.log(1 + data_trades[i]['ReturnTradedPrice'])
     #data_trades[i]= data_trades[i].ix[1:] #--check this works before you run it pls



#print raw_data.head(10)
print 'Did the First Bit'

ticker_list=data_trades[0]['#RIC'].unique()
for ticker_ in ticker_list:
    if not os.path.exists(_path+ticker_):
        os.makedirs(_path+ticker_)

date_list =data_trades[0]['Date[G]'].unique()
#print(ticker_list)
#
for j in range(no_files):
    for ticker_ in ticker_list:
        root=(_path + ticker_)
        ticker_data[ticker_] = data_trades[j][data_trades[j]['#RIC']==ticker_]
        #ticker_data[ticker_].to_csv(_path+ ticker_+ '.csv')
        for date_ in date_list:
            date_data[date_]=ticker_data[ticker_][ticker_data[ticker_]['Date[G]'] == date_]
            #date_data[date_]['Date[G]'].apply(_date_convert)
            date_data[date_].to_csv(root + '/' +date_ +'.csv')

            #date_data[date_].dropna()


print 'Finished. check your directory'
print date_data[date_list[0]].head(10)



#


