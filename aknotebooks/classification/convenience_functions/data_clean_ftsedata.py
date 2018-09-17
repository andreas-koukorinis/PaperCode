#need these columns: ['TradedTime', 'Duration','TradedPrice', 'ReturnTradedPrice']
# #_df_
import pandas as pd
import numpy as np
import sys#need these columns: ['TradedTime', 'Duration','TradedPrice', 'ReturnTradedPrice']
# #_df_

import datetime
from datetime import date, datetime, timedelta
import time
import os, fnmatch
import numpy as np
import shutil
ftse_folder = ('/home/ak/Documents/Data/FTSE/')
usb_folder= ('/media/ak/My Passport/FTSE/unzipped/FTSE/')
### make a list of all the files, this will be used to parse through ###
### ignore files that are there before ###

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

def time_in_millis(_timestamp):
    return time.mktime(pd.Timestamp(pd.to_datetime(_timestamp)).timetuple())

def file_name(_date_):
    t= pd.to_datetime(_date_)
    t_ =t.to_datetime()
    t_.date()
    _file_name_ = t_.strftime('%Y%m%d') #proper format for the HMM code '20180301'
    return _file_name_

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
        print('New Directory Created',file_path)
    else:
        print('Directory Exists!check')
def match_files(_path,_string):
    _files=fnmatch.filter(os.listdir(_path), str(_string))
    return _files

def symbol_path_(_symb):
    _symbol_path = os.path.join(ftse_folder,_symb+'/')
    print(_symbol_path)
    ensure_dir(_symbol_path)
    return _symbol_path

####main code###
#os.listdir(usb_)
list_= os.listdir(usb_folder)
symbols=[]
for _index, _ in enumerate(list_):
    print list_[_index][40:43]
    symbols.append(list_[_index][40:43])

for _symbol_ in symbols:
    print "working on Symbol:", _symbol_
    _files=fnmatch.filter(os.listdir(usb_folder), '*'+_symbol_+'*.csv')
    _symbol_path= symbol_path_(_symbol_)

#pick the first file from list/this in general will be 1
#better way to write this
    for index_, _ in enumerate(_files):
        _file_ =pd.read_csv(usb_folder+_files[index_])
        _dates_= _file_[_file_['Type']=='Trade']['Date[G]'].unique() #make a list of the unique dates to parse through
        _trades_file_=_file_[_file_['Type']=='Trade'][['Date[G]','Time[G]','Price','Volume']]
        for _no_,_ in enumerate(_dates_):
            print _dates_[_no_]
            _date_file_ = _trades_file_[_trades_file_['Date[G]']==_dates_[_no_]]
            _df_=_date_file_.rename(index=str, columns={'Time[G]':'TradedTime', 'Price':'TradedPrice'})
            _millis_ = _df_['TradedTime'].apply(time_in_millis)
            _df_['Duration']=_millis_.diff().dropna() #duration of trades
            _df_['ReturnTradedPrice']=_df_['TradedPrice'].pct_change()
            _final_=_df_[1:] #drop first line as there are NaNs and really does not fit
            print "saving file/date:", _dates_[_no_]
            _final_.to_csv(_symbol_path+file_name(_dates_[_no_])+'.csv')


