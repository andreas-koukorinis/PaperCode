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
import clfutils
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
        self.jointLocsSymbols = list(np.unique([f.split("_")[0] for f in os.listdir(self.jointLocationsPickleFolder)]))

    def return_pickled_dict(self):
        # returns the filename of the joint features and labels file
        # the features file is a dictionary that has keys
#         if self.symbol in self.jointLocsSymbols:
        if self.alternate_label_idx < 4:
            pickle_in_filename_local = os.path.join(self.jointLocationsPickleFolder, "_".join(
                (self.symbol, self.LabelsAlternateName[self.alternate_label_idx], 'FeaturesLocations.pkl')))
        else:
            print('Error in the alternate label index: value between 0 and 4')
#         else:
#             print('Symbol is not in the folder')
        return pickle_in_filename_local

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

    def ma_spread(self, short_window=5, long_window=10):
        # function that produces the MA spread, which can be used on its own or as an input for MACD
        short_rolling_px = self.df['TradedPrice'].rolling(window=short_window).mean()
        long_rolling_px = self.df['TradedPrice'].rolling(window=long_window).mean()
        px_name = "_".join(('px_indx', str(short_window), str(long_window)))
        self.df[px_name] = long_rolling_px - short_rolling_px
        return self.df

    def ma_spread_duration(self, short_window=2, long_window=4):
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
    # just pick symbols I have joint locations
    jointLocsSymbols = list(np.unique([f.split("_")[0] for f in os.listdir(jointLocationsPickleFolder)]))

    symbol = 'BATS.L'
    best_svc_dict = defaultdict(dict)
    forward_dates_dict =  defaultdict(dict)
    if symbol in jointLocsSymbols:
        print('ok to go')
        alternate_label_idx = 1
        # pick a label indexprint(jointLocsSymbols[symbol_idx], ' and labels ',
        # labels_pickle_files[alternate_label_idx])
        symbol_idx = jointLocsSymbols.index(symbol)  # dont particularly need this!
        print(symbol, ' and labels ', labels_pickle_files[alternate_label_idx])
        data_cls = AlternateLabelFeaturesLoader(path_main=dataDrive, symbol=jointLocsSymbols[symbol_idx],
                                                alternate_label_idx=alternate_label_idx,
                                                jointLocationsPickleInput=jointLocationsPickleFolder)
        jointLocationsDictionary = (data_cls.load_pickled_in_filename(data_cls.return_pickled_dict()))
        joint_keys = data_cls.joint_loc_pickle_keys(data_cls.return_pickled_dict())
        logmemoryusage("Before garbage collect")
        gc.collect()  # continue
        for joint_key_idx, joint_key_date in enumerate(joint_keys):
            # this is a date - and we will enumerate through the keys
            # getting features and labels
            logmemoryusage("Before feature creation")
            features, labels = ticker_features_labels(jointLocationsDictionary[joint_keys[joint_key_idx]])
            print(joint_key_date)
            label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
            features_df = hmm_features_df(
                features)  # features data-frame - this just unbundles the features into a dataframe
            # lets get all the features in order now#
            market_features_df = CreateMarketFeatures(
                CreateMarketFeatures(CreateMarketFeatures(df=CreateMarketFeatures(df=labels).ma_spread_duration())
                                     .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
            df_concat = pd.DataFrame(pd.concat([features_df, market_features_df], axis=1, sort='False').dropna())
            df = df_concat[df_concat[label_name].notna()]
            df_final = df.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                        'Volume', label_name])
            y_labels_train = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]
            if df_final.shape[0] < 10:
                print(' the ratio of classes is too low. try another label permutation')
                continue
            else:
                X_train = MinMaxScaler().fit_transform(df_final)
                models_cls = clfutils.FitModels(X_train, y_labels_train)
                best_svc_dict[symbol][joint_key_date] = {'SVC': models_cls.best_svc_clf()}
                forward_dates_dict[symbol][joint_key_date] = data_cls.forwardDates(joint_keys, joint_key_date)

            pickle_out_filename = os.path.join(dataDrive, "_".join(
                (symbol, labels_pickle_files[alternate_label_idx], 'SingleKernelSVC.pkl')))
            pickle_out = open(pickle_out_filename, 'wb')
            pickle.dump(best_svc_dict, pickle_out)
            pickle_out.close()

            forward_dates_dict_filename = os.path.join(dataDrive, "_".join(
                (symbol, labels_pickle_files[alternate_label_idx],joint_key_date ,'ForwardDates.pkl')))
            forward_dates_dict_out = open(forward_dates_dict_filename, 'wb')
            pickle.dump(best_svc_dict, forward_dates_dict_out)
            forward_dates_dict_out.close()



