import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

sc = StandardScaler()
import os
import pickle
import fnmatch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score


##useful function


def fwd_dates(_dates_list, _key_date):
    # returns a list of dates that are forward from the key_date
    fwd_dates_list = [i for i in _dates_list if i > _key_date]
    return fwd_dates_list


def remove_nans(features_tuple, labels, idx=1):
    # not the cleanest but useful
    # function to clean up nans as I seem to use it a lot, so better to have one function
    # combines the features and labels and removes rows with nans across so we dont lose the ordering
    # returns features and labels
    features_df = pd.concat([features_tuple[0], features_tuple[1], features_tuple[2], \
                             features_tuple[3]], axis=1, sort=False)
    labels_only = labels.drop(columns=['ReturnTradedPrice', 'Duration', 'states', 'TradedTime',
                                       'TradedPrice'], axis=1)
    df_concat = pd.concat([features_df, labels_only.iloc[:, 0:idx]], axis=1, sort='False')
    # only using 1st set of labels- but we can re-write this a bit
    df_x_nan = df_concat.dropna()  # dropping all nans
    label_column_loc_ = df_x_nan.shape[1] - 1  # location of labels column in the clean df
    labels_ = df_x_nan.iloc[:, label_column_loc_:label_column_loc_ + 1]  # keep pure labels
    features_ = df_x_nan.drop(df_x_nan.columns[label_column_loc_], axis=1)  # keeping the features only
    return features_, labels_


def prec_recall_report(y_true, y_predict):
    # function to ge the sci-kit learn classification metrics into a pretty DF for csv!
    report = pd.DataFrame(list(precision_recall_fscore_support(y_true, y_predict)),
                          index=['Precision', 'Recall', 'F1-score', 'Support']).T
    # Now add the 'Avg/Total' row
    report.loc['Avg/Total', :] = precision_recall_fscore_support(y_true, y_predict, average='weighted')
    report.loc['Avg/Total', 'Support'] = report['Support'].sum()
    return report


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

    def load_data(self):
        pass

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


data_dir = os.getenv('FINANCE_DATA')  # main directory referenced in all the code\n",
data_only_drive = '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2'  # external date only drive\n",

"# location to save results\n",
model_save_loc = os.path.join(data_only_drive, 'Data', 'features_models', 'models')
