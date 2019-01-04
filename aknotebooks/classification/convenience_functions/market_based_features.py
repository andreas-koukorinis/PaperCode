import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import BDay
import pickle
import fnmatch
import datetime as dt
import glob
def listdirs(folder): #return only directories from a master folder
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def find_path(basepath, stringname):
    for fname in os.listdir(basepath):
        path = os.path.join(basepath, fname)
        if os.path.isdir(path):
            if stringname in fname:
                return path
def get_ticket(rx_path):
    ticker=rx_path.split(".")[0].split("/")[-1].split("-")[0]
    return ticker

def ticker_trades_dir(ticker):
    dest=os.path.join(os.getenv('FINANCE_DATA'), "_".join((ticker,'trades')))
    if not os.path.isdir(dest):
        os.makedirs(dest)

def quotes_trades_dir(ticker):
    dest=os.path.join(os.getenv('FINANCE_DATA'), "_".join((ticker,'quotes')))
    if not os.path.isdir(dest):
        os.makedirs(dest)

def mkt_features_dir(symbol):
    ##going to save all the price based features in the data only drive as everything needs to be moved there anyway
    features_models = os.path.join(data_only_drive,'Data','features_models')
    dest=os.path.join(features_models,symbol,'MARKET_BASED')
    if not os.path.isdir(dest):
        print('dir does not exist- making directory:',dest)
        os.makedirs(dest)
    else:
        print('dir does exist:')

def agg_on_trd_time(gr):
    """
    Utility func to aggregate trades on timestamp. All trades with equal time stamp
    will collapse to one row and the traded price will be the volume weighted traded
    price.
    """
    vTrdPrice = np.sum(gr['TradedPrice'] * gr['Volume'])/np.sum(gr['Volume'])
    volume = np.sum(gr['Volume'])

    return pd.Series({'Volume': volume,
                      'TradedPrice': vTrdPrice})


class MarketFeatures(object):
    # a class to be expanded that uses features for base case -market based only-indicators/features
    """"Requires:
    a dataframe that has TradedPrice And Volume columns
    symbol - A stock symbol on which to form a strategy on.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average.
    """

    def __init__(self, symbol, df):
        self.symbol = symbol
        self.df = df

    def load_data(self):
        pass

    def ma_spread(self, short_window=5, long_window=20):
        # function that produces the MA spread, which can be used on its own or as an input for MACD
        short_rolling_px = self.df['TradedPrice'].rolling(window=short_window).mean()
        long_rolling_px = self.df['TradedPrice'].rolling(window=long_window).mean()
        px_name = "_".join(('px_indx', str(short_window), str(long_window)))
        self.df[px_name] = long_rolling_px - short_rolling_px
        return self.df

    def obv_calc(self, df):
        # on balance volume indicator
        self.df['SignedVolume'] = self.df['Volume'] * np.sign(self.df['TradedPrice'].diff()).cumsum()
        self.df['SignedVolume'][:1] = 0
        self.df['OBV'] = self.df['SignedVolume'].cumsum()
        self.df = df.drop(columns=['SignedVolume'])
        return self.df

    def chaikin_mf(self, df, period=5):
        # chaiking money flow indicator
        df["MF Multiplier"] = (self.df['TradedPrice'] - (self.df['TradedPrice'].expanding(period).min()) \
                               - (self.df['TradedPrice'].expanding(period).max() \
                                  - self.df['TradedPrice'])) / (
                                          self.df['TradedPrice'].expanding(period).max() - self.df[
                                      'TradedPrice'].expanding(period).min())
        self.df["MF Volume"] = self.df['MF Multiplier'] * df['Volume']
        self.df['CMF_' + str(period)] = self.df['MF Volume'].sum() / self.df["Volume"].rolling(period).sum()
        self.df = self.df.drop(columns=['MF Multiplier', 'MF Volume'])
        return self.df


if __name__ == '__main__' :

    #  setting various locations ##

    data_dir = os.getenv('FINANCE_DATA')  # main directory
    features_models = os.path.join(data_dir,'features_models')
    features = os.path.join(features_models,'features')
    models = os.path.join(features_models,'models')
    labels = os.path.join(features_models,'labels')
    data_only_drive = '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2' ##data only external##

    # get list of FTSE symbols
    ftse_symbols = [s for s in os.listdir(features) if s.endswith('.L')]

    for symbol in ftse_symbols:
        print('doing symbol:', symbol)
        # look in the DataOnly drive
        features_models = os.path.join(data_only_drive, 'Data', 'features_models')

    # precise location is the main_directory : inside of features_models, we have a per symbol market based directory

        main_dir=os.path.join(features_models, symbol, 'MARKET_BASED')
    # get a list of all the files from the original directory of the symbol
        list_inputs=os.listdir(os.path.join(data_dir, symbol))
        for idx, input_date in enumerate(list_inputs):
            # just a couple of stupid print statements
            print('doing date:', input_date)
            print('file input:', os.path.join(data_dir, symbol,list_inputs[idx]))
            # read the data form the original directory
            df = pd.read_csv(os.path.join(data_dir, symbol, list_inputs[idx]))
            # calculate all the features- this can be done a tad cleaner
            market_features=MarketFeatures(df=df, symbol=symbol)
            df_mf = market_features.ma_spread(short_window=3, long_window=7)
            df_mf = market_features.obv_calc(df_mf)
            df_mf = market_features.chaikin_mf(df_mf, period=5) #5 trade money flow
            df_mf = df_mf.drop(columns=['TradedPrice', 'ReturnTradedPrice', 'Volume', 'Duration']) #take out non features
            # destination of the market features data_frame: data drive, with the date of the original input file
            df_mf.to_csv(os.path.join(main_dir, input_date))
