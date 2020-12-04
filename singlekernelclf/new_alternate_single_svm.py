import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import fileutils as fileutils
import gc
import psutil
import pickle
import fnmatch
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from collections import defaultdict

# paths
dataDrive = '/media/ak/My Passport/Data/FinDataReal/'  # also labels location folder
jointLocationsPickleFolder = os.path.join(dataDrive, 'JointLocationsDicts')
extPath = '/media/ak/My Passport/ExperimentData'
featuresPath = "/".join((extPath, 'features'))  # path with features

# Labels
labels_location_folder = fileutils.data_path  # this is the folder where all the labels are saved

labels_pickle_files = [s for s in os.listdir(labels_location_folder) if ('LabelsAlternate') in s if
                       not ('.pkl') in s]  # these are all the dicts that we have alternate labels for.
# labels_pickle_files: these are all the dicts that we have alternate labels for.

symbols = [f for f in [s for s in os.listdir(labels_location_folder) if '.L' in s if '_Features' not in s] if
           ('.L_A' or '_Features') not in f]  # from all


def logmemoryusage(msg):
    # function to log memory usage
    process = psutil.Process(os.getpid())
    print('Memory usage at %s is %smb.' % (msg, process.memory_info().rss / 1000 / 1000))


def unpickle_csv(pickled_csv):
    with open(pickled_csv, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    return p


def ticker_features_labels(file_joint_locations):
    # input a joint location file that contains both features and labels and returns one
    labels = pd.read_csv(file_joint_locations[1])
    features = unpickle_csv(file_joint_locations[0])
    return features, labels


def hmm_features_df(features_tuple):
    return pd.concat([features_tuple[0], features_tuple[1], \
                      features_tuple[2], features_tuple[3]], axis=1, sort=False)


class AlternateLabelFeaturesLoader(object):
    """
    takes in a main path, a symbol, an alternate label index (from 0 to 4) and returns, the pickled dict file name
    and path for the common locations
    """

    def __init__(self, path_main, symbol, alternate_label_idx=None,
                 jointLocationsPickleInput=jointLocationsPickleFolder):
        self.main_path = path_main
        self.symbol = symbol
        self.LabelsAlternateName = ['LabelsAlternateFive', 'LabelsAlternateFour', 'LabelsAlternateOne',
                                    'LabelsAlternateThree', 'LabelsAlternateTwo']
        self.alternate_label_idx = alternate_label_idx
        self.jointLocationsPickleFolder = jointLocationsPickleInput

    def return_pickled_dict(self):
        # returns the filename of the joint features and labels file
        # the features file is a dictionary that has keys
        if self.alternate_label_idx < 4:
            pickle_in_filename = os.path.join(self.jointLocationsPickleFolder, "_".join(
                (self.symbol, self.LabelsAlternateName[self.alternate_label_idx], 'FeaturesLocations.pkl')))
        else:
            print('Error in the alternate label index: value between 0 and 4')
        return pickle_in_filename

    @staticmethod
    def load_pickled_in_filename(file):
        # load a simple pickled file and return it. its a bit different to the method used for the dictionary as this
        # is pure Python 3.x
        pickle_in = open(file, 'rb')
        return pickle.load(pickle_in)

    @staticmethod
    def joint_loc_pickle_keys(inputFile):
        # returns keys of joint locations from labels and features
        return list(AlternateLabelFeaturesLoader.load_pickled_in_filename(inputFile).keys())

    @staticmethod
    def forwardDates(list_of_keys, current_date):
        """ return all the forward looking dates for each idxKey we use for training"""
        lookAheadKeys = sorted(i for i in list_of_keys if i > current_date)
        return lookAheadKeys


class CreateMarketFeatures(object):
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


if __name__ == "__main__":
    symbol_idx = 1  # pick a symbol
    alternate_label_idx = 1  # pick a label index
    print(symbols[symbol_idx], ' and labels ', labels_pickle_files[alternate_label_idx])
    data_cls = AlternateLabelFeaturesLoader(path_main=dataDrive, symbol=symbols[symbol_idx],
                                            alternate_label_idx=alternate_label_idx,
                                            jointLocationsPickleInput=jointLocationsPickleFolder)
    jointLocationsDictionary = (data_cls.load_pickled_in_filename(data_cls.return_pickled_dict()))
    joint_keys = data_cls.joint_loc_pickle_keys(data_cls.return_pickled_dict())

    logmemoryusage("Before garbage collect")
    gc.collect()  # continue
    joint_key_idx = 0  # this is a date - and we will enumerate through the keys
    # getting features and labels
    features, labels = ticker_features_labels(jointLocationsDictionary[joint_keys[joint_key_idx]])
    features_df = hmm_features_df(features)  # features data-frame
    market_features_df = CreateMarketFeatures(
        CreateMarketFeatures(CreateMarketFeatures(df=CreateMarketFeatures(df=labels).ma_spread_duration())
                             .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
    df_concat = pd.DataFrame(pd.concat([features_df, market_features_df], axis=1, sort='False').dropna())

    logmemoryusage("Before feature creation")

    label_name = str(df_concat.columns[df_concat.columns.str.contains(pat='label')].values[0])

    df_final = df_concat.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'Unnamed: 0', 'ReturnTradedPrice',
                                       'Volume', label_name])
    if df_final.shape[0] > 10:
        print('sorting out features and labels')

        X_train = MinMaxScaler().fit_transform(df_final)

        y_labels = df_concat[df_concat.columns[df_concat.columns.str.contains(pat='label')]].iloc[:, 0]

        # else:
    #     print('basically if we have too many fucking nans')

#
#
# forward_dates_keys = data_cls.forwardDates(joint_keys, joint_keys[joint_key_idx])  # forward dates for this date
