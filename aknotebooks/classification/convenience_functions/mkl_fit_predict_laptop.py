from MKLpy.preprocessing import normalization, rescale_01
from sklearn.metrics.pairwise import rbf_kernel as RBF
import os
import pandas as pd
import numpy as np
import pickle as pkl
from MKLpy.metrics.pairwise.misc import homogeneous_polynomial_kernel as HPK_kernel

np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.multiclass import OneVsRestClassifier  # support from multiclass

import time
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.model_selection import cross_val_score, cross_val_predict
import pickle as pkl

from MKLpy.metrics import pairwise
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc



def forwardDates(idxKey, commonLocs):
    """ return all the forward looking dates for each idxKey we use for training"""
    keys = list(commonLocs.keys())  # this produces a list with 2 keys -first HMM Symbol Date and common Date
    lookAheadKeys = sorted(i for i in keys if i > keys[idxKey])
    return dict((k, commonLocs[k]) for k in lookAheadKeys)


if __name__ == '__main__':

    # paths
    extPath = '/media/ak/My Passport/ExperimentData'
    featuresPath = "/".join((extPath, 'features'))
    labelsPath = "/".join((extPath, 'labels'))
    MKLPath = "/".join((extPath, 'MKLExpPath'))
    symbolsList = sorted(list(set(os.listdir(featuresPath)) & set(os.listdir(labelsPath))))
    symbolsToRemove = ['AAL.L', 'BLT.L', 'BARC.L', 'AZN.L', 'CNA.L', 'APF.L','CCL.L','CPG.L','LAND.L','RTO.L','RSA.L','RDSb.L','PSON.L']
    symbols = [x for x in symbolsList if x not in symbolsToRemove]

    # symbol specific

    symbolIdx = 27# pick a symbol -1,2,4,7,9,10,12,15,17,19, 22, 26, 27
    for symbolIdx, _ in enumerate(symbols):
        symbol = symbols[symbolIdx]
        print(symbol)

        MKLExpPath = "/".join((MKLPath, symbol, 'Experiments'))  # symbol experiment path
        MKLSymbolPath = "/".join((MKLPath, symbol))  # symbol path
        MKLKernPath = "/".join((MKLPath, symbol, 'Kernels'))  # symbol kernel path
        print(symbol)
        os.listdir(MKLSymbolPath)

        ''' specific symbol targets'''

        symbolFeaturesLocation = "/".join((featuresPath, symbol, 'MODEL_BASED'))
        symbolLabelsLocation = "/".join((labelsPath, symbol, 'NON_DIRECTIONAL'))  # location of labels for each symbol
        symbolHMMDatesList = os.listdir(symbolFeaturesLocation)
        # list of all the MODEL dates we have generated features files for. each #
        # each of these dates in symbolFeaturesDates corresponds to a list of dates
        # (symbolHMMDatesList = '20170829', '20170710', '20170801', ... ]
        symbolLabelsDates = [dateFile.split(".")[0] for dateFile in os.listdir(symbolLabelsLocation)]

        ''' keep track of various structs'''
        # now lets go down into each HMM-model date, and pick all the forward futures (out of sample)
        hmmFeatureLocations = {}  # symbol-hmm-model-date index --> this is the indexation in symbolFeaturesDatesList
        commonDatesDict = {}  # this is a struct that will contain for each HMM date, the common labels/features- this
        # should
        # be used for training and testing
        createDate = []  # place holder for the hash key of when the features got created
        symbolEachModelFeaturesDates = {}
        HMMModelFeaturesLabelsCommon = {}  # location dictionary with 2 keys: HMM Date and Common Date
        commonDates = []

        LocDictsList = []  # symbol specific

        ''' specific symbol targets'''

        for hmmDateIdx, hmmDate in enumerate(sorted(symbolHMMDatesList)):
            symbolModelFeaturesDate = os.path.join(symbolFeaturesLocation, symbolHMMDatesList[hmmDateIdx])
            if len(os.listdir(symbolModelFeaturesDate)) != 0:
                create_date = os.listdir(symbolModelFeaturesDate)[0].split("_")[-2]
                # output looks like this: /media/ak/DataOnly/FinDataReal/PRU.L/MODEL_BASED/20170710
                symbolEachModelFeaturesDates[symbolHMMDatesList[hmmDateIdx]] = [file.split("_")[5] for file in
                                                                            os.listdir(symbolModelFeaturesDate)]
                # output is a dictionary where the keys are the HMM models dates and the values a list of dates - for
                # each HMM date we have a list of features
                for keyHMMDate in sorted(list(symbolEachModelFeaturesDates.keys())):  # for each of the HMM model dates
                    print(keyHMMDate)
                    common_dates = list(set(symbolEachModelFeaturesDates[keyHMMDate]) & set(symbolLabelsDates))
                    # take the list of feature dates (conditional on HMM model date) + the list of labels -intersection!

                    '''we now produce a dict for each HMM model, where each value is a list of common dates and we are key-ed by
                    the HMM Date '''
                    commonDatesDict[keyHMMDate] = common_dates
                    for commonDate in common_dates:
                        '''iterate through all the common dates and figure out the location of each file for labels and
                        features '''
                        labelsCommonFileLoc = "/".join((symbolLabelsLocation, ".".join((commonDate, 'csv'))))
                        #             comnDateFeatureLocMaster = os.path.join((symbolModelFeaturesDate, commonDate))
                        commonDatesFeatureFile = "".join(
                            (
                            symbols[symbolIdx], '_3_states_features_date:_', commonDate, "_now:_", create_date, "_.pickle"))
                        FeatureFileLoc = os.path.join(symbolModelFeaturesDate, commonDatesFeatureFile)
                        conditions = [os.path.exists(FeatureFileLoc), os.path.exists(labelsCommonFileLoc)]

                        if all(conditions):
                            print('all good on Date:', commonDate)
                            HMMModelFeaturesLabelsCommon[keyHMMDate, commonDate] = [FeatureFileLoc, labelsCommonFileLoc]
                            pkl.dump(HMMModelFeaturesLabelsCommon,
                                     open("/".join((MKLSymbolPath, "LocDictsListCorrect.pkl")), "wb"))

                        else:
                            print('problem on date: ', commonDate)
                            continue
                        print('saving:')
                        pkl.dump(commonDatesDict,
                                 open("/".join((MKLSymbolPath, "CommonLocationsDicts.pkl")), "wb"))
            else:
                continue
