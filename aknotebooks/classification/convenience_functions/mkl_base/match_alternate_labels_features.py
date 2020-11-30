import pandas as pd
import os
import numpy as np
import pickle as pkl
import fileutils as fileutils
import collections
from collections import defaultdict
from itertools import *
import gc
import pickle

"""
Code to match alternative labels to features and save a dictionary that has all the locations for both to be used in model generation
Produces a dictionary back that has all the locations for each HMM and common date
"""

LabelsAlternateNames = \
    ['LabelsAlternateOne', 'LabelsAlternateTwo', 'LabelsAlternateThree', 'LabelsAlternateFour',
     'LabelsAlternateFive']  # names of alternate labels

# paths that are needed for this work
dataDrive = '/media/ak/My Passport/Data/FinDataReal/'
extPath = '/media/ak/My Passport/ExperimentData'
featuresPath = "/".join((extPath, 'features'))  # path with features
labels_location_folder = fileutils.data_path  # this is the folder where all the labels are saved

labels_pickle_files = [s for s in os.listdir(labels_location_folder) if ('LabelsAlternate') in s if
                       not ('.pkl') in s]  # these are all the dicts that we have alternate labels for.
# labels_pickle_files: these are all the dicts that we have alternate labels for.

symbols = [f for f in [s for s in os.listdir(labels_location_folder) if '.L' in s if '_Features' not in s] if
               ('.L_A' or '_Features') not in f]  # from all
if __name__ == '__main__':

    for alternate_label_idx,_ in enumerate(LabelsAlternateNames): # iternate through all the labels
        symbols_with_alternate_labels = os.listdir( os.path.join(labels_location_folder, LabelsAlternateNames[alternate_label_idx])) # get a list of all the symbols with alternate labels

        symbols_with_features = os.listdir(featuresPath)  # not sure I need these!
        # get a subset of the features that we know work for both symbols and labels
        symbols_with_features_and_labels = sorted([x for x in symbols_with_alternate_labels if x in symbols_with_features])

        symbol_idx = 3  # pick a symbol idx
        for symbol_idx,_ in enumerate(symbols_with_features_and_labels):
            symbol = symbols_with_features_and_labels[symbol_idx]  # pick a symbol
            print(symbol)
            symbolFeaturesLocation = os.path.join(featuresPath, symbol, 'MODEL_BASED')
            symbolHMMDatesList = os.listdir(symbolFeaturesLocation)

            symbolFeaturesLocation = "/".join((featuresPath, symbol, 'MODEL_BASED'))
            # symbolLabelsLocation = ## need to think how I deal with this one

            # list of all the MODEL dates we have generated features files for. each #
            # each of these dates in symbolFeaturesDates corresponds to a list of dates
            # (symbolHMMDatesList = '20170829', '20170710', '20170801', ... ]
            symbolLabelsDates = os.listdir(os.path.join(dataDrive, LabelsAlternateNames[alternate_label_idx],
                                                        symbol))  # true dates of data labels --> which correspond for all the dates
            # lay out the paths for each date
            symbolLabelsDatesPaths = {
                f: os.path.join(dataDrive, LabelsAlternateNames[alternate_label_idx], symbol, f, f + '.csv') for f in
                symbolLabelsDates}

            # now for each HMM Date, you get a set of features, train a model, then run that model to all possible dates!
            # and move on

            # now lets go down into each HMM-model date, and pick all the forward futures (out of sample)
            symbolEachHMMFeatureLocations = {}  # symbol-hmm-model-date index --> this is the indexation in symbolFeaturesDatesList
            commonDatesDict = {}  # this is a struct that will contain for each HMM date, the common labels/features- this
            # should
            # be used for training and testing
            createDate = []  # place holder for the hash key of when the features got created
            symbolEachModelFeaturesDates = {}
            HMMModelFeaturesLabelsCommon = {}  # location dictionary with 2 keys: HMM Date and Common Date
            commonDates = []
            symbolEachHMMFeaturesLocationsDict = {}  # what it says: create Features Locations for each HMM

            LocDictsList = []  # symbol specific

            for hmmDateIdx, hmmDate in enumerate(sorted(symbolHMMDatesList)):
                symbolModelFeaturesDate = os.path.join(symbolFeaturesLocation,
                                                       symbolHMMDatesList[hmmDateIdx])  # location of the features_files

                if len(os.listdir(symbolModelFeaturesDate)) != 0:
                    # produce two dictionaries - one with dates and one with locations- all indexed by the same HMM model
                    # date
                    symbolEachModelFeaturesDates[symbolHMMDatesList[hmmDateIdx]] = [file.split("_")[5] for file in
                                                                                    os.listdir(symbolModelFeaturesDate)]
                    symbolEachHMMFeatureLocations[symbolHMMDatesList[hmmDateIdx]] = [
                        os.listdir(os.path.join(symbolFeaturesLocation, symbolHMMDatesList[hmmDateIdx]))]
            symbolEachModelFeaturesDatesKeys = sorted(list(symbolEachModelFeaturesDates.keys()))
            symbolEachHMMFeatureLocationsKeys = sorted(list(symbolEachHMMFeatureLocations.keys()))

            print(symbolEachHMMFeatureLocationsKeys)
            symbolEachHMMFeaturesLocationsDict = {symbolEachModelFeaturesDatesKeys[idx]: dict(
                zip(sorted(symbolEachModelFeaturesDates[symbolEachModelFeaturesDatesKeys[idx]]),
                    sorted(symbolEachHMMFeatureLocations[symbolEachModelFeaturesDatesKeys[idx]][0]))) for idx, _ in
                enumerate(symbolEachModelFeaturesDatesKeys)}
            # '''we now produce a dict for each HMM model, where each value is a list of common dates and we are key-ed by
            #                    the HMM Date '''
            commonDatesDict = {keyHMMDate: list(set(symbolEachModelFeaturesDates[keyHMMDate]) & set(symbolLabelsDates)) for
                               keyHMMDate in sorted(list(symbolEachModelFeaturesDates.keys()))}

            joint_labels_features_dict = {}

            for commonDate in list(commonDatesDict.keys()):
                #     print('common date',os.path.join(symbolFeaturesLocation, commonDate))
                #     commonDateLabelPath = symbolLabelsDatesPaths[commonDate]
                #     print(symbolEachHMMFeaturesLocationsDict[commonDate].keys())
                for key in list(symbolEachHMMFeaturesLocationsDict[commonDate].keys()):
                    print('-------produce a model:')
                    print(key)
                    print(symbolEachHMMFeaturesLocationsDict[commonDate][key])
                    print(symbolLabelsDatesPaths[key])
                    print('--------------')
                    is_valid_features_file = os.path.join(symbolFeaturesLocation, commonDate,
                                                          symbolEachHMMFeaturesLocationsDict[commonDate][key])
                    print(os.path.isfile(is_valid_features_file))
                    print('--------------')
                    print('is valid labels file?')
                    print(os.path.isfile(symbolLabelsDatesPaths[key]))
                    print('---------------------')
                    joint_labels_features_dict[key] = [is_valid_features_file, symbolLabelsDatesPaths[key]]

                pickle_out_filename = os.path.join(dataDrive, "_".join(
                    (symbol, LabelsAlternateNames[alternate_label_idx], 'FeaturesLocations.pkl')))
                pickle_out = open(pickle_out_filename, 'wb')
                pickle.dump(joint_labels_features_dict, pickle_out)
                pickle_out.close()
