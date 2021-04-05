import os
import pickle
import pandas as pd


'''
written in April 2021, for final set of experiments and results for single kernel CLFs
'''


class DataLoader(object):
    def __init__(self, path_main, symbol):
        self.main_path = path_main
        self.symbol = symbol

        self.symbol_features_path = os.path.join(self.main_path, str('SymbolFeatureDirectories'), self.symbol , 'MODEL_BASED')
        # collection of per symbol non directional labels - this will give the location of all the sub-directories
        self.main_labels_path = os.path.join(self.main_path, 'ExperimentCommonLocs')
        # this will give the location of all the alternative labels files
        #
        # # list of all the model -oos hmm feature dates - each folder is a collection of oos feature dates
        self.hmm_dates_list = os.listdir(self.symbol_features_path)
        # each folder are the OOS features from each HMM

        # directory with all the symbols that have features at the moment
        self.all_symbols_features_paths = os.path.join(self.main_path, 'SymbolFeatureDirectories')

    def compute_date(self):
        """
        :param symbol_idx: symbol we are looking to get the compute date for
        :return: compute date
        """
        compute_date = os.listdir(self.symbol_features_path)[1]#.split("_")[7]
        return compute_date

    # def symbol_specific_label_path(self, label_number):
    #     """
    #     gets for each symbol label number combination the specific path
    #     :param label_number: takes value 1 to 7 in numeric format
    #     :return: returns a path
    #     """
    #     labels_numbers = {1: 'LabelsAlternateOne', 2: 'LabelsAlternateTwo', 3: 'LabelsAlternateThree',
    #                       4: 'LabelsAlternateFour', 5: 'LabelsAlternateFive', 6: 'LabelsAlternateSix',
    #                       7: 'LabelsAlternateSeven'}
    #     return os.path.join(self.main_labels_path, labels_numbers[label_number], self.symbol)
    #
    # def symbol_specific_labels_dates(self, label_number):
    #     """
    #
    #     :param label_number: label number and takes value 1 to 7
    #     :return: list of dates
    #     """
    #     list_of_dates = [f.split(".")[0] for f in os.listdir(self.symbol_specific_label_path(label_number))]
    #     return list_of_dates
    #
    # def ticker_features(self, model_date, date):
    #     # need to make this a lot more flexible with number of states
    #     if model_date < date:
    #         file_name = "_".join(
    #             (self.ticker, '3', 'states', 'features', 'date:', date, 'now:', self.compute_date, '.pickle'))
    #         file_loc = os.path.join(self.symbol_features_path, str(model_date), file_name)
    #         with open(file_loc, 'rb') as handle:
    #             ticker_features = pickle.load(handle)
    #     else:
    #         print('Loading Feature Date which is in-sample. Change your Model Date')
    #
    #     return ticker_features
    #
    # def ticker_labels_csv(self, label_number, date):
    #     file_loc = os.path.join(self.symbol_specific_label_path(label_number), str(date) + '.csv')
    #     ticker_labels = pd.read_csv(file_loc, index_col=0)
    #     return ticker_labels
    #
    # @staticmethod
    # def open_pickle_file(path, pickle_file):
    #     file_loc = os.path.join(path, pickle_file)
    #     pickle_to_file = pickle.load(open(file_loc, "rb"))
    #     return pickle_to_file
    #
    # @staticmethod
    # def get_date_from_file(file_, numb_):
    #     return os.path.splitext(file_[numb_])[0]
    #
    # # ######### Locations of various files ######
    # #
    # # labels_location_files = ''  # where all the labels are stored
    # # mainPath = '/media/ak/DataOnly/SecondAlternativeExperimentPath/'  # all results and various other files will be stored here
    # #
    # # # all the features are stored here in symbol directories - so you need os.path.join(symbol_feature_directors, [Symbol], 'MODEL_BASED')
    # # # # and this will result in subfolders of dates
    # # symbol_feature_directories = '/media/ak/DataOnly/SymbolFeatureDirectories/'
    # #
    if __name__ == "__main__":
        from singlekernelclf.all_symbols_clfs_fit import DataLoader

        # test = DataLoader('/media/ak/DataOnly/', 'AAL.L')
        # #print(test.compute_date())
        # print(test.symbol_features_path)
        # print(test.symbol_specific_label_path(2))
