from __future__ import division
import pickle
import os
import pandas as pd

import numpy as np
import time

# filelocations
clean_raw_data_path = '/media/ak/DataOnly/Data/'  # has a list of symbols in here with processed data-appendix is a list of symbols
data_path = '/media/ak/DataOnly/FinDataReal/'
labels_path = os.path.join(data_path, 'Labels')
# list of symbols directory that has 2 files - this basically has two folders, one for HMM and one for SVM
singleSVMmodels_path = '/media/ak/DataOnly/Data/features_models/models/'
altFeatures_path = '/media/ak/WorkDrive/Data/features_models/features/'


# todo: list of symbols for all the various features dates- think about how this is used in general though? do i need both

# for features, you append the data_path with a symbol and you get the list of dates for all the features
# alternative way to load: '/media/ak/DataOnly/Data/features_models/models/', 'clfs_model_hash.pickle')

def no_nans(label):
    return np.sum(np.isnan(label))


def remove_last_element(arr):
    return arr[np.arange(arr.size - 1)]


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)

    # check length
    if len(a_set.intersection(b_set)) > 0:
        return a_set.intersection(b_set)
    else:
        return "no common elements"


def get_files_for_ticker(ticker):
    """
    :return: Return all file names available for ticker, sort by name which is date
    """
    fin_data_dir = os.path.join("/media/ak/DataOnly", 'Data')
    total_path = os.path.join(fin_data_dir, ticker)
    # os.path.join(total_path, f)
    files_for_ticker = [f.split('.')[0] for f in os.listdir(total_path)
                        if (os.path.isfile(os.path.join(total_path, f)) and '.csv' in f and 'lock' not in f)]

    files_for_ticker.sort()
    return files_for_ticker


class DataLoader(object):
    """
    class that loads and cleans up data
    constructor params: path_main
    todo: needs some refactoring so i dont produce the same function 2x and i can read from predecessors
    """

    def __init__(self, path_main, ticker):
        self.main_path = path_main
        self.ticker = ticker

        # collection of per symbol non directional labels
        self.symbol_labels_path = os.path.join(self.main_path, 'Labels', self.ticker, 'NON_DIRECTIONAL')
        self.symbol_features_path = os.path.join(self.main_path, self.ticker, 'MODEL_BASED')
        # list of all the model -oos hmm feature dates - each folder is a collection of oos feature dates
        self.hmm_dates_list = os.listdir(self.symbol_features_path)  # each folder are the OOS features from each HMM
        self.compute_date = os.listdir(os.path.join( \
            self.symbol_features_path, \
            os.listdir(self.symbol_features_path)[1]))[1].split("_")[7]

    def symbol_features_date_load(self, hmm_model_date, features_date):
        """
        hmm_model_date: the hmm model that was used to produce the features, it is hashed by the date
        features_date: the features date that we want to retrieve the features for
        return: set of features for 3 state hmm for hmm_model_date, features_date combination
        """
        # need to make this a lot more flexible with number of states
        if hmm_model_date < features_date:  # condition needed to make sure we dont pick-insample data
            featuresfilename = "_".join(
                (self.ticker, '3', 'states', 'features', 'date:', str(features_date), 'now:', self.compute_date,
                 '.pickle'))
            file_loc = os.path.join(self.symbol_features_path, str(hmm_model_date), featuresfilename)

            with open(file_loc, 'rb') as handle:
                ticker_features_date = pickle.load(handle, encoding='latin1')

        if hmm_model_date > features_date:
            raise ValueError('Loading Feature Date which is in-sample. Change your Model Date')

        return ticker_features_date

    def symbol_labels_date(self, date):
        """
        takes a specific date that labels exist and returns the exact labels from that dataframe
        parameter: date that labels exist

        """
        file_loc = os.path.join(self.symbol_labels_path, str(date) + '.csv')
        symbol_date_df = pd.read_csv(file_loc, index_col=0)
        return symbol_date_df.filter(like='label', axis=1)

    def symbol_date_processed_data_load(self, date):
        """
        takes a specific date that labels exist and returns the processed data frame of raw data
        parameter: date that labels exist

        """
        return pd.read_csv(os.path.join(self.symbol_labels_path, str(date) + '.csv'))

    @staticmethod
    def open_pickle_file(path, pickle_file):
        file_loc = os.path.join(path, pickle_file)
        pickle_to_file = pickle.load(open(file_loc, "rb"), encoding='latin1')
        return pickle_to_file

    @staticmethod
    def get_date_from_file(file_, numb_):
        return os.path.splitext(file_[numb_])[0]
