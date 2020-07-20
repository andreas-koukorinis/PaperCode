from __future__ import division
import pickle
import os
import pandas as pd
import numpy as np
import time


class MarketFeatures(object):
    # a class to be expanded that uses features for base case -market based only-indicators/features
    """"Requires:
    a dataframe that has TradedPrice And Volume columns
    symbol - A stock symbol on which to form a strategy on.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average.
    """

    def __init__(self, df):
        #         self.ticker = ticker
        self.df = df

    def ma_spread(self, short_window=5, long_window=20):
        # function that produces the MA spread, which can be used on its own or as an input for MACD
        short_rolling_px = self.df['TradedPrice'].rolling(window=short_window).mean()
        long_rolling_px = self.df['TradedPrice'].rolling(window=long_window).mean()
        px_name = "_".join(('px_indx', str(short_window), str(long_window)))
        self.df[px_name] = long_rolling_px - short_rolling_px
        return self.df

    def ma_spread_duration(self, short_window=5, long_window=20):

        # function that produces the MA spread, which can be used on its own or as an input for MACD

        short_rolling_px = self.df['Duration'].rolling(window=short_window).mean()
        long_rolling_px = self.df['Duration'].rolling(window=long_window).mean()
        dur_name = "_".join(('dur_indx', str(short_window), str(long_window)))
        self.df[dur_name] = long_rolling_px - short_rolling_px
        return self.df

    def obv_calc(self):
        # on balance volume indicator
        self.df['SignedVolume'] = self.df['Volume'] * np.sign(self.df['TradedPrice'].diff()).cumsum()
        self.df['SignedVolume'].iat[1] = 0
        self.df['OBV'] = self.df['SignedVolume']  # .cumsum()
        self.df = self.df.drop(columns=['SignedVolume'])
        return self.df

    def chaikin_mf(self, period=5):

        # Chaikin money flow indicator
        self.df["MF Multiplier"] = (self.df['TradedPrice'] - (self.df['TradedPrice'].expanding(period).min()) \
                                    - (self.df['TradedPrice'].expanding(period).max() \
                                       - self.df['TradedPrice'])) / (
                                           self.df['TradedPrice'].expanding(period).max() - self.df[ \
                                       'TradedPrice'].expanding(period).min())
        self.df["MF Volume"] = self.df['MF Multiplier'] * self.df['Volume']
        self.df['CMF_' + str(period)] = self.df['MF Volume'].sum() / self.df["Volume"].rolling(period).sum()
        self.df = self.df.drop(columns=['MF Multiplier', 'MF Volume'])
        return self.df
