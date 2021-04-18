from __future__ import division
import pickle
import os
import pandas as pd

import numpy as np
import time


# Written in April 2021, this is basically a clean up version of the file I need to use for loading data.

# # ######### Locations of various files ######

def paths(path_name):
    '''
    :param path_name: short cut name for paths
    :return: path URL
    '''
    pathDict = {'experiments': '/media/ak/DataOnly/SecondAlternativeExperimentPath/',
                'symbols_features': '/media/ak/DataOnly/SymbolFeatureDirectories/',
                'main': '/media/ak/DataOnly/'}
    return pathDict[path_name]


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


class DataStorage(object):
    """
    class that stores data of all sorts
    """

    def __init__(self, path_main, symbol):
        self.main_path = path_main
        self.symbol = symbol

    def check_if_path_exists_or_create(self, sub_path):
        """ take a subpath and check if it exists, if not create it"""

        if not os.path.isdir(os.path.join(self.main_path, sub_path, self.symbol)):
            os.makedirs(os.path.join(self.main_path, sub_path, self.symbol))

    @staticmethod
    def store_processed_data_pickle(pickle_out_filename, file_to_pickle):
        pickle_out = open(pickle_out_filename, 'wb')
        pickle.dump(file_to_pickle, pickle_out)
        pickle_out.close()
        return print("file stored:", pickle_out_filename)

    def storage_common_locations(self, hmm_features_list):
        """

        :param symbol: symbol that we have locations of Data that is clean, i.e matched labels and features
        :return: full absolute path
        """
        storage_location = os.path.join(self.main_path, 'MKLCommonDataLocations', self.symbol)
        return storage_location

    def storage_fitted_model_locations(self):
        """

        :param symbol: symbol that we have locations for fitted models
        :return: full absolute path
        """
        storage_location = os.path.join(self.main_path, 'MKLFittedModels', self.symbol)
        return storage_location

    def storage_fitted_model_locations(self):
        """

        :param symbol: symbol that we have locations for fitted models
        :return: full absolute path
        """
        storage_location = os.path.join(self.main_path, 'MKLOOSPredictions', self.symbol)
        return storage_location

#
class DataLoader(object):
    """
    class that loads and cleans up data
    constructor params: path_main
    todo: needs some refactoring so i dont produce the same function 2x and i can read from predecessors
    """

    def __init__(self, path_main, symbol):
        self.main_path = path_main
        self.symbol = symbol
        self.symbol_features_path = os.path.join(self.main_path, str('SymbolFeatureDirectories'), self.symbol,
                                                 'MODEL_BASED')
        # collection of per symbol non directional labels - this will give the location of all the sub-directories
        self.main_labels_path = os.path.join(self.main_path, 'ExperimentCommonLocs')
        # this will give the location of all the alternative labels files

        # list of all the model -oos hmm feature dates - each folder is a collection of oos feature dates
        self.hmm_dates_list = os.listdir(self.symbol_features_path)
        # each folder are the OOS features from each HMM

        # directory with all the symbols that have features at the moment
        self.all_symbols_features_paths = os.path.join(self.main_path, 'SymbolFeatureDirectories')

    def compute_date(self):
        """
        :param symbol_idx: symbol we are looking to get the compute date for
        :return: compute date
        """
        compute_date = \
            os.listdir(os.path.join(self.symbol_features_path, os.listdir(self.symbol_features_path)[1]))[1].split("_")[
                -2]
        return compute_date

    def symbol_specific_label_path(self, label_number):
        """
        gets for each symbol label number combination the specific path
        :param label_number: takes value 1 to 7 in numeric format
        :return: returns a path
        """
        labels_numbers = {1: 'LabelsAlternateOne', 2: 'LabelsAlternateTwo', 3: 'LabelsAlternateThree',
                          4: 'LabelsAlternateFour', 5: 'LabelsAlternateFive', 6: 'LabelsAlternateSix',
                          7: 'LabelsAlternateSeven'}
        return os.path.join(self.main_labels_path, labels_numbers[label_number], self.symbol)

    def symbol_specific_labels_dates(self, label_number):
        """

        :param label_number: label number and takes value 1 to 7
        :return: list of dates
        """
        list_of_dates = [f.split(".")[0] for f in os.listdir(self.symbol_specific_label_path(label_number))]
        return list_of_dates

    def ticker_features(self, model_date, date):
        # TODO: need to make this a lot more flexible with number of states
        """
        this loads up a particular feature date
        """
        if model_date < date:
            file_name = "_".join(
                (self.symbol, '3', 'states', 'features', 'date:', date, 'now:', str(self.compute_date()), '.pickle'))
            file_loc = os.path.join(self.symbol_features_path, str(model_date), file_name)
            with open(file_loc, 'rb') as handle:
                ticker_features = pickle.load(handle, encoding='latin1')
        else:
            print('Loading Feature Date which is in-sample. Change your Model Date')

        return ticker_features

    def ticker_specific_label_date_csv(self, label_number, date):
        """
        returns specific label number and date file for ticker. has data and labels
        """
        file_loc = os.path.join(self.symbol_specific_label_path(label_number), str(date) + '.csv')
        ticker_labels = pd.read_csv(file_loc, index_col=0)
        return ticker_labels

    @staticmethod
    def open_pickle_file(path, pickle_file):
        file_loc = os.path.join(path, pickle_file)
        pickle_to_file = pickle.load(open(file_loc, "rb"), encoding='latin1')
        return pickle_to_file

    @staticmethod
    def get_date_from_file(file_, numb_):
        return os.path.splitext(file_[numb_])[0]

    def hmm_model_date_feature_list_filepaths(self, hmm_date):
        """
        produces a list of HMM model- feature dates combinations
        :param hmm_date: put a date that an HMM exists
        :return: get a list of paths for each of the HMM dates
        """
        feature_dates_files = [os.path.join(self.symbol_features_path, hmm_date, f)
                               for f in os.listdir(os.path.join(self.symbol_features_path, hmm_date))]

        features_dates_files_dict = {f.split("_")[5]: os.path.join(self.symbol_features_path, hmm_date, f) for f in
                                     os.listdir(os.path.join(self.symbol_features_path, hmm_date))
                                     if os.path.isfile(os.path.join(self.symbol_features_path, hmm_date, f))}
        # decided to return both so I can have the key in the second case to merge them with the
        # output of the function below and have all my files in the same place

        return [sorted(feature_dates_files), features_dates_files_dict]

    def hmm_model_feature_corrsp_labels_files(self, hmm_date, label_idx):
        """
        picks up from what we did above so for each HMM dates, strips out the features dates and produces labels files where they exist only
        :param hmm_date: get an HMM date
        :param label_idx: get a label ID from 1-7
        :return: all the various labels files! the keys will be used to construct out of sample dates!
        """
        features_dates_files = [f.split("_")[6] for f in self.hmm_model_date_feature_list_filepaths(hmm_date)[0]]
        label_path = self.symbol_specific_label_path(label_idx)
        feature_corresponding_labels_paths = {f: os.path.join(label_path, str(f) + '.csv') for f in features_dates_files
                                              if os.path.isfile(os.path.join(label_path, str(f) + '.csv'))}

        return feature_corresponding_labels_paths


if __name__ == '__main__':
    import os

    print(sorted(os.listdir(paths('symbols_features'))))
