
import pickle
import os
import pandas as pd
import numpy as np
import seaborn as sns
import time

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

###

from tqdm import tqdm

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

import sys

sys.path.insert(0, '/directory/tothe/handshakefile/')
import os

from scipy.stats import norm, ttest_ind

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataFolder = '/media/ak/WorkDrive/Data'
figures_destination = '/home/ak/Documents/Research/Papers/figures/'

import matplotlib as mpl

plt.style.use(os.path.join(mpl.get_configdir(), 'latexstyle.mplstyle'))

import math

import scipy as sp
import scipy
import scipy.optimize as opt
import scipy.stats as stats

import xlwt
import logging


def logger(name, level=logging.INFO):
    tmp_logger = logging.getLogger(name)
    if not tmp_logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(name)s  %(levelname)s %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        tmp_logger.setLevel(level)
        tmp_logger.addHandler(handler)
    return tmp_logger


class volatilityEstimation(object):

    def __init__(self, df, tick_window=20, clean=True):
        self.df = df
        self.tick_window = tick_window
        self.mu_1 = np.sqrt((2 / np.pi))
        self.mu_43 = 2 ** (2 / 3) * scipy.special.gamma(7 / 6) ** scipy.special.gamma(1 / 2) ** (-1)
        self.clean = True  # set to true to remove nans at the beginning of the period
        self.trading_seconds_btc = 86400
        self.trading_seconds_futures = 23400

        self.column_open = 'micro_price_open'
        self.column_high = 'micro_price_high'
        self.column_low = 'micro_price_low'
        self.column_close = 'micro_price_close'

        self.log_hl = (df[str(self.column_high)] / df[str(self.column_low)]).apply(np.log)
        self.log_co = (df[str(self.column_close)] / df[str(self.column_open)]).apply(np.log)
        # median sampling frequency

        self.z = pd.DataFrame(
            [((x.hour * 60 + x.minute) * 60 + x.second) for x in self.df['TimeStamp_open']]).diff().fillna(0).astype(
            'float64')
        self.clean_arrival_rates = np.asarray(
            self.z[(self.z > self.z.quantile(0.003)) & (self.z < self.z.quantile(0.97))].dropna()).astype('float64')
        # clean arrival rates after i remove all the outliers
        self.mean_sampling_frequency = np.mean(np.asarray(self.clean_arrival_rates))

    def arrival_rates(self):

        z = pd.DataFrame([((x.hour * 60 + x.minute) * 60 + x.second) for x in self.df['TimeStamp_open']]).diff().fillna(
            0).astype('float64')
        clean_arrival_rates = np.asarray(z[(z > z.quantile(0.003)) & (z < z.quantile(0.97))].dropna()).astype('float64')
        # clean arrival rates after i remove all the outliers

        return clean_arrival_rates

    def M_parameter(self):
        """
        normalised sampling frequency for adjustments
        """

        M_btc = self.trading_seconds_btc / self.mean_sampling_frequency
        M_futures = self.trading_seconds_futures / self.mean_sampling_frequency

        return [M_btc, M_futures]

    def garmanKlass(self, trading_periods):

        rs = 0.5 * self.log_hl ** 2 - (2 * np.log(2) - 1) * self.log_co ** 2

        def f(v):
            return (trading_periods * v.mean()) ** 0.5

        result = rs.rolling(window=self.tick_window, center=False).apply(func=f)

        if self.clean:
            return result.dropna()
        else:
            return result

    def parkinson(self, trading_periods, clean=True):

        rs = (1.0 / (4.0 * np.log(2.0))) * (self.df[str(self.column_high)] / self.df[str(self.column_close)]).apply(
            np.log) ** 2.0

        def f(v):
            return trading_periods * v.mean() ** 0.5

        result = rs.rolling(window=self.tick_window, center=False).apply(func=f)

        if self.clean:
            return result.dropna()
        else:
            return result

    def bipower_variation(self, rollingWindow=5):
        '''
        Bipower Variation (BV) is the sum of the product of absolute time series returns
        :param column: price column
        :return: returns bivariate variation
        Barnhorf - Nielse & Shephard 2004 & 2006
        '''
        bv = self.mu_1 ** (-2) * (
            (self.log_hl.abs() * self.log_hl.shift(1).abs()).fillna(0).rolling(rollingWindow).sum())

        if self.clean:
            return bv.dropna()
        else:
            return bv

    def tripower_quarticity(self, rollingWindow=5, sampling_param=0):
        """
        using M for BTC here - change sampling param to 1 for everything else
        """
        tpq = self.M_parameter()[sampling_param] * self.mu_43 ** (-3) * ((self.log_hl.abs() ** (
                4 / 3) * self.log_hl.shift(1).abs() ** (4 / 3) * self.log_hl.shift(2).abs() ** (4 / 3)).rolling(
            rollingWindow).sum().fillna(0))

        if self.clean:

            return tpq.dropna()
        else:
            return tpq

    def realised_variance(self, rollingWindow=5):
        """
        realised variance - andersen & bollerselv -1998
        """

        realvar = (self.log_hl ** 2).rolling(rollingWindow).sum().fillna(0)

        if self.clean:
            return realvar.dropna()
        else:
            return realvar

    def realised_absolute_variation(self, rollingWindow=5, sampling_param=0):
        """
        realised absolute variation - Forsberg & Ghysels 2007
        using M for BTC here - change sampling param to 1 for everything else
        """

        realabsovar = self.mu_1 ** (-2) * self.M_parameter[sampling_param] ** (-0.5) * self.log_hl.rolling(
            rollingWindow).sum().fillna(0)

        return realabsovar

    def realised_skewness_kurtosis(self, rollingWindow=15, sampling_param=0):

        """
        set the rolling window to 10-15
        sampling param for BTC ~ 0
        """

        rm3 = (self.log_hl ** 3).rolling(rollingWindow).sum().fillna(0)
        rm4 = (self.log_hl ** 4).rolling(rollingWindow).sum().fillna(0)

        rs = np.sqrt(self.M_parameter()[sampling_param]) * (
                rm3 / self.realised_variance(rollingWindow=rollingWindow)) ** (3 / 2)
        rk = self.M_parameter()[sampling_param] * (rm4 / self.realised_variance(rollingWindow=rollingWindow)) ** 2

        return rs, rk

    def jumps_test(self, rollingWindow, sampling_param=0):
        """
        Jump test by Huang and Tauchen 2005
        adopted by repo: RealisedQuantities
        """

        j1 = (np.log(self.realised_variance(rollingWindow=rollingWindow)) - np.log(
            self.bipower_variation(rollingWindow=rollingWindow)))
        j2 = (((self.mu_1 ** -4) + 2 * (self.mu_1 ** -2) - 5) /
              (self.M_parameter()[0] * self.tripower_quarticity(rollingWindow=rollingWindow,
                                                                sampling_param=sampling_param)
               * (self.bipower_variation(rollingWindow=rollingWindow) ** -2)))

        j = j1 / (j2 ** 0.5)

        return (j.abs() >= stats.norm.ppf(0.995)) * 1  # can also look at 0.999)*

    def relative_jump_measure(self, rollingWindow, sampling_param=0):

        nominator = (self.realised_variance(rollingWindow=rollingWindow)
                     - self.bipower_variation(rollingWindow=rollingWindow))

        measure_to_return = nominator / self.bipower_variation(rollingWindow=rollingWindow)

        return measure_to_return

    def negative_semivariance(self, x):
        """
        input: returns
        negative realised semivariance
        Barndorff- Nielsen
        """
        return x.apply(lambda x: (x ** 2)) * (x < 0).sum()

    def positive_semivariance(self, x):
        """
        input: returns
        positive realised semivariance
        Barndorff- Nielsen
        """
        return x.apply(lambda x: (x ** 2)) * (x > 0).sum()

    def signed_jump_variation(self, x):

        sjv = self.positive_semivariance(x) - self.negative_semivariance(x)
        sjv_p = sjv * (sjv > 0)
        sjv_n = sjv * (sjv < 0)

        return sjv_p, sjv_n

    def continuous_discontinuous_quadratic_variance(self):

        """
        seperate continuous and discontinuous parts of quadratic variation

        """
        pass


class descriptiveStats(object):

    def __init__(self, df):
        self.df = df
        self.column_open = 'micro_price_open'
        self.column_high = 'micro_price_high'
        self.column_low = 'micro_price_low'
        self.column_close = 'micro_price_close'
        self.log_hl = (df[str(self.column_high)] / df[str(self.column_low)]).apply(np.log)
        self.log_co = (df[str(self.column_close)] / df[str(self.column_open)]).apply(np.log)

    def base_descriptive_stats(self):
        desc_stats = dict()
        desc_stats['median_hl'] = np.median(self.log_hl)
        desc_stats['mean_co'] = np.median(self.log_co)
        desc_stats['quantile_95_hl'] = self.log_hl.quantile(0.95)
        desc_stats['quantile_95_co'] = self.log_co.quantile(0.95)
        desc_stats['q3_75_hl'] = self.log_hl.quantile(0.75)
        desc_stats['q3_75_co'] = self.log_co.quantile(0.75)
        desc_stats['q1_25_hl'] = self.log_hl.quantile(0.25)
        desc_stats['q1_25_co'] = self.log_co.quantile(0.25)
        desc_stats['outlier_lower_limit_hl'] = desc_stats['q1_25_hl'] - 1.5 * (
                desc_stats['q3_75_hl'] - desc_stats['q1_25_hl'])
        desc_stats['outlier_upper_limit_hl'] = desc_stats['q3_75_hl'] + 1.5 * (
                desc_stats['q3_75_hl'] - desc_stats['q1_25_hl'])
        desc_stats['lower_limit_outliers'] = self.log_hl[self.log_hl > desc_stats['outlier_lower_limit_hl']].count()
        desc_stats['upper_limit_outliers'] = self.log_hl[self.log_hl > desc_stats['outlier_upper_limit_hl']].count()
        desc_stats['total_outliers'] = desc_stats['lower_limit_outliers'] + desc_stats['upper_limit_outliers']

        return desc_stats


class DataLoader:
    """
    loads data and constructs full LOB and calendar bars for each one
    """

    def __init__(self, data_location,
                 symbol,
                 dates,
                 use_columns,
                 calendar_resample,
                 trade_volume_width,
                 ticks_width,
                 usd_volume_width):

        self._data_location = data_location
        self._symbol = symbol
        self._input_dates = dates
        self._use_columns = use_columns
        self._calendar_resample = calendar_resample
        self._ticks_width = ticks_width
        self._trade_volume_width = trade_volume_width
        self._usd_volume_width = usd_volume_width
        self._dates = []
        self._bars_dict = {}
        self.calendar_bars = []
        self.volume_bars = []
        self.tick_bars = []
        self.usd_volume_bars = []
        self._logger = logger('DataLoader')
        #         self._symbols = sorted(['VIX',  'FB1',  'TU1',  'G_1',  'RX1',  'OE1',  'TY1',  'FV1',  'JB1',  'RX1',  'DU1',  'KE1',  'US1',  'YM1', 'XM1',  'VXX'] )
        self._data_folder = data_location

        self._quotes_string = "".join((self._symbol, '_Comdty_quotes'))
        self._trades_string = "".join((self._symbol, '_Comdty_trades'))

        #         self._quotes_files =sorted([ f for f in os.listdir(self._data_folder) if str('_quotes') in f])
        #         self._trades_files = sorted([f for f in os.listdir(self._data_folder) if str('_trades') in f])

        self._quotes_Files = os.path.join(self._data_folder, self._quotes_string)
        self._trades_Files = os.path.join(self._data_folder, self._trades_string)

    def load_and_format_data(self):
        mergedFile = dict()
        for date in self._input_dates:
            dateToLoad = os.path.join(date + '.csv')
            self._logger.info(f"Loading data for:{date}")

            quotesDateFile = os.path.join(self._quotes_Files, dateToLoad)
            print(quotesDateFile)
            tradesDateFile = os.path.join(self._trades_Files, dateToLoad)
            print(tradesDateFile)

            try:
                qt_tmp_df = pd.read_csv(quotesDateFile, usecols=self._use_columns)
                qt_tmp_df['TradeTime'] = pd.to_datetime(qt_tmp_df.time).values

                quotes_df = pd.merge_asof(
                    qt_tmp_df[qt_tmp_df['type'] == 'BID'].dropna().fillna("ffill").sort_values('TradeTime')
                    , qt_tmp_df[qt_tmp_df['type'] == 'ASK'].dropna().fillna("ffill").sort_values('TradeTime'),
                    on='TradeTime', allow_exact_matches=True)

                quotes_df = quotes_df.rename(columns={'value_x': 'BestBid', 'value_y': 'BestAsk',
                                                      'size_x': 'BidSize', 'size_y': 'AskSize', 'time_x': 'QuoteTime'})

                tr_tmp_df = pd.read_csv(tradesDateFile, usecols=self._use_columns)
                tr_tmp_df['TradeTime'] = pd.to_datetime(tr_tmp_df.time).values
                trades_df = tr_tmp_df.dropna().fillna("ffill").sort_values('TradeTime')
                trades_df = trades_df.rename(columns={'value': 'TradePrice',
                                                      'size': 'TradeSize'})
                trades_columns = ['TradeSize', 'type', 'TradePrice', 'TradeTime']

                LOB = pd.merge_asof(quotes_df, trades_df[trades_columns].sort_values('TradeTime'), on='TradeTime',
                                    allow_exact_matches=True)

                LOB.BidSize = LOB['BidSize'].replace(0, 1)
                LOB.AskSize = LOB['AskSize'].replace(0, 1)
                LOB['TimeStamp'] = pd.to_datetime(LOB.TradeTime).dt.time
                LOB['TradeVolume'] = LOB['TradeSize'].fillna(0)
                LOB['total_traded_volume'] = LOB.TradeVolume
                LOB['milliSeconds'] = [(((x.hour * 60 + x.minute) * 60 + x.second) * 1000) for x in LOB['TimeStamp']]
                LOB['dollar_traded_volume'] = pd.Series(LOB.TradePrice * LOB.TradeVolume).fillna(0)
                LOB['timeStampIdx'] = pd.DatetimeIndex(LOB.time_y)
                LOB['micro_price'] = (LOB.BestAsk * LOB.AskSize + LOB.BestBid * LOB.BidSize) / (
                        LOB.AskSize + LOB.BidSize)  #

                mergedFile[date] = LOB

            except FileNotFoundError:
                self._logger.info(f"Data for day {date} does not exist. Skipping this day")
            return mergedFile

    @staticmethod
    def prep_bars(df, drop_col):
        df.columns = [f"{i[0]}_{j}" for i, j in zip(df.columns, ['open', 'high', 'low', 'close'] * len(df.columns))]
        if drop_col:
            df.drop([f"{drop_col}_{i}" for i in ['open', 'high', 'low', 'close']], axis=1, inplace=True)
        return df

    def drop_lob_columns(df):
        # not using this anywhere at the moment

        ask_price_columns_to_be_dropped = list(pd.DataFrame(df.filter(like='ask_price_', axis=1)).columns.values)
        ask_size_columns_to_be_dropped = list(pd.DataFrame(df.filter(like='ask_size_', axis=1)).columns.values)
        bid_size_columns_to_be_dropped = list(pd.DataFrame(df.filter(like='bid_size_', axis=1)).columns.values)
        bid_price_columns_to_be_dropped = list(pd.DataFrame(df.filter(like='bid_price_', axis=1)).columns.values)
        time_stamps = ['timestamp', 'timestamp.1', 'exchange_time', 'feed_time']
        columns_to_be_dropped = ask_price_columns_to_be_dropped + ask_size_columns_to_be_dropped + \
                                bid_size_columns_to_be_dropped + bid_price_columns_to_be_dropped + \
                                time_stamps
        df = df.drop(columns=columns_to_be_dropped)

        return df

    def get_bars(self, df):

        self._logger.info('Calculating calendar clock')
        calendar_bars = df.resample(rule=self._calendar_resample, on='timeStampIdx').agg(
            ['first', 'max', 'min', 'last'])
        calendar_bars = self.prep_bars(calendar_bars, 'timeStampIdx')

        self._logger.info('Calculating volume clock')
        # TODO: Doing this on day-by-day basis. We lose the last turn of the clock each day
        df['cum_total_trade_volume'] = df.total_traded_volume.cumsum()
        df['volume_bucket'] = np.floor(df.cum_total_trade_volume / self._trade_volume_width)
        volume_bars = df.groupby(by='volume_bucket').agg(['first', 'max', 'min', 'last']).iloc[1:-1]
        volume_bars = self.prep_bars(volume_bars, 'cum_total_trade_volume')
        df.drop(['cum_total_trade_volume', 'volume_bucket'], axis=1, inplace=True)

        self._logger.info('Calculating tick clock')
        # TODO: ticks here means any change to the order book. Not necessarily a trade. Can add that clock too
        df['tick_count'] = range(len(df))
        df['tick_bucket'] = np.floor(df.tick_count / self._ticks_width)
        tick_bars = df.groupby(by='tick_bucket').agg(['first', 'max', 'min', 'last']).iloc[1:-1]
        tick_bars = self.prep_bars(tick_bars, 'tick_count')
        df.drop(['tick_count', 'tick_bucket'], axis=1, inplace=True)

        self._logger.info('Calculating usd volume clock')
        df['cum_usd_volume'] = df.dollar_traded_volume.cumsum()
        df['usd_volume_bucket'] = np.floor(df.cum_usd_volume / self._usd_volume_width)
        usd_volume_bars = df.groupby(by='usd_volume_bucket').agg(['first', 'max', 'min', 'last']).iloc[1:-1]
        usd_volume_bars = self.prep_bars(usd_volume_bars, 'cum_usd_volume')
        df.drop(['cum_usd_volume', 'usd_volume_bucket'], axis=1, inplace=True)

        self.calendar_bars.append(calendar_bars)
        self.volume_bars.append(volume_bars)
        self.tick_bars.append(tick_bars)
        self.usd_volume_bars.append(usd_volume_bars)

        self._bars_dict['calendar_bars'] = self.calendar_bars
        self._bars_dict['volume_bars'] = self.volume_bars
        self._bars_dict['tick_bars'] = self.tick_bars
        self._bars_dict['usd_volume_bars'] = self.usd_volume_bars

    def get_all_dates_bars(self, input_dict):
        output_dict = {}
        for input_date in self._input_dates:
            output_dict[input_date] = self.get_bars(input_dict[input_date])

        return output_dict

    @staticmethod
    def get_concat_data(input_dict):
        concat_dict = dict()
        for bar in input_dict.keys():
            concat_dict[bar] = pd.concat([input_dict[bar][i]
                                          for i in range(len(input_dict[bar]))], ignore_index=False)
        return concat_dict


def return_scaled(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.array(x).reshape(-1, 1))
    return np.asarray(x_scaled)


####
def symbol_file_name_path(experimentsDestination, symbol, file_idx):
    symbolPath = os.path.join(experimentsDestination, str(symbol))
    file_names = sorted(os.listdir(symbolPath))
    file_name = file_names[file_idx]
    file_path = os.path.join(symbolPath, file_name)
    return file_names, file_path


def n_F(dicts, bar):
    df = pd.DataFrame.from_dict(dicts[str(bar)])
    n = df.index.values
    F = df.median(axis=1).values
    return n, F


def h_params(h_dict, bar):
    h_df = pd.DataFrame.from_dict(h_dict[str(bar)]).T.median(axis=0)
    H_intercept = h_df[1]
    H = h_df[0]
    return H_intercept, H


def open_pickle_filepath(pickle_file):
    # open pickle filepath
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file


def returns(s):
    # compute log returns
    arr = np.diff(np.log(s))
    return pd.Series(arr, index=s.index[1:])


def df_derived_by_shift(df, lag=0, NON_DER=[]):
    df = df.copy()
    if not lag:
        return df
    cols = {}
    for i in range(1, lag + 1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i += 1
        df = pd.concat([df, dfn], axis=1)
    return df


def get_test_stats(bar_types, bar_returns, test_func, *args, **kwds):
    dct = {bar: (int(bar_ret.shape[0]), test_func(bar_ret, *args, **kwds))
           for bar, bar_ret in zip(bar_types, bar_returns)}
    df = (pd.DataFrame.from_dict(dct)
          .rename(index={0: 'sample_size', 1: f'{test_func.__name__}_stat'})
          .T)
    return df


def apply_micro_structure_features(df):
    print("Applying micro-structure features")
    df = df.reset_index(level=0)

    df['weighted_average_BestBid'] = pd.DataFrame(df.filter(like='BestBid', axis=1).values).mul(
        pd.DataFrame(df.filter(like='BidSize', axis=1).values)).sum(axis=1) / pd.DataFrame(
        df.filter(like='BidSize', axis=1).values).sum(axis=1)

    df['weighted_average_BestAsk'] = pd.DataFrame(df.filter(like='BestAsk', axis=1).values).mul(
        pd.DataFrame(df.filter(like='AskSize', axis=1).values)).sum(
        axis=1) / pd.DataFrame(df.filter(like='AskSize', axis=1).values).sum(axis=1)
    df['weighted_activity_spread'] = df['weighted_average_BestAsk'] - df['weighted_average_BestBid']
    df['total_size'] = (pd.DataFrame(df.filter(like='AskSize', axis=1).values).sum(axis=1) +
                        pd.DataFrame(df.filter(like='BidSize', axis=1).values).sum(axis=1))
    df['micro_price'] = ((pd.DataFrame(df.filter(like='BestBid', axis=1).values).mul(
        pd.DataFrame(df.filter(like='BidSize', axis=1).values)).sum(axis=1) +
                          pd.DataFrame(df.filter(like='BestAsk', axis=1).values).mul(
                              pd.DataFrame(df.filter(like='AskSize', axis=1).values)).sum(axis=1))) / df[
                            'total_size']
    df['price_imbalance'] = ((pd.DataFrame(df.filter(like='BestAsk', axis=1).values).mul(
        pd.DataFrame(df.filter(like='AskSize', axis=1).values)).sum(axis=1) -
                              pd.DataFrame(df.filter(like='BestBid', axis=1).values).mul(
                                  pd.DataFrame(df.filter(like='BidSize', axis=1).values)).sum(axis=1))) / df[
                                'total_size']
    df['pct_change_micro_price'] = df.micro_price_close.pct_change()
    df['simple_mid_price'] = 0.5 * (pd.DataFrame(df.filter(like='BestAsk', axis=1)).mean(axis=1) +
                                    pd.DataFrame(df.filter(like='BestBid', axis=1)).mean(axis=1))

    # TODO: do we want to be using pct_change here or log returns?
    return df


# locations
elements = '/media/ak/Elements/OrderBookProcessedData'
experimentsLocation = '/media/ak/WorkDrive/Data/July2nd2022_Experiments'

# params
symbols = sorted(os.listdir(elements))
bars = ['volume_bar', 'calendar_bar', 'usd_volume_bar', 'tick_bar']
