#%%

import MKLpy
import pandas as pd
import numpy as np
import sklearn.svm as svm
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier  # support from multiclass
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt
from MKLpy.metrics import pairwise
from MKLpy.preprocessing import normalization, rescale_01

from sklearn.metrics.pairwise import rbf_kernel
#MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD#KOMD is not a MKL algorithm but a simple kernel machine like the SVM
from MKLpy.model_selection import cross_val_score, cross_val_predict
scaler = StandardScaler()

dataOnlyDrive = '/media/ak/DataOnly'  # external date only drive
data_dir = os.getenv('FINANCE_DATA') #internal folder with finance data
folderList = [s for s in os.listdir(dataOnlyDrive) if s.startswith('Dat') or s.startswith('Fin')]

folderIdx = 0  # pick a folder- essentially where all the financial data we will use are stored
finalLocation= "/".join((dataOnlyDrive,folderList[folderIdx]))
symbols = [s for s in os.listdir(finalLocation) if s.endswith('.L')]  #keep a list of the symbols

##picking up a specific symbol
symbolIdx = 1 #pick one of the symbols
#symbols[symbolIdx] -->output :PRU.L

print(symbols[symbolIdx])

symbolLocation = "/".join((finalLocation,symbols[symbolIdx])) # do a join to get the location

# get he features now
symbolFeaturesLocation = "/".join((symbolLocation,'MODEL_BASED'))

#  where all the HMM output is -but I am picking this symbol and reading from this folder:')

symbolHMMDatesList = os.listdir(symbolFeaturesLocation)

# list of all the HMM MODEL dates we have generated features files for. each #
# each of these dates in symbolFeaturesDates corresponds to a list of dates
# (symbolFeaturesDatesList = '20170829', '20170710', '20170801', ... ]
# now lets move to labels ...pick the location of the labels and make a list of all the available labels #

# location of labels : /media/ak/DataOnly/FinDataReal/Labels/[Symbol :PRU.L]/NON_DIRECTIONAL

symbolLabelsLocation = "/".join((finalLocation, 'Labels',symbols[symbolIdx],'NON_DIRECTIONAL'))

#list of all the label dates

symbolLabelsDates =[dateFile.split(".")[0] for dateFile in os.listdir(symbolLabelsLocation)]

print('symbolLabelsDates', symbolLabelsDates)
                    # Output symbolLabelsDates --> ['20170704', '20180226', '20180208',...] all we are doing is going for this

# now lets go down into each HMM-model date, and pick all the forward futures (out of sample)

hmmFeatureLocations = {}  # symbol-hmm-model-date index --> this is the indexation in symbolFeaturesDatesList
commonDatesDict = {}  # this is a struct that will contain for each HMM date, the common labels/features- this should be used for training and testing
createDate = []  # place holder for the hash key of when the features got created
commonFeaturesLocs = {}
commonLabelsLocs = {}
commonLocs = {}

# this symbolFeaturesDatesList[featrsIdx] will give you a date: 20170710 =which contains all the HMM induced featureb

for hmmDateIdx, hmmDate in enumerate(np.sort(symbolHMMDatesList)):

    symbolModelFeaturesDate = "/".join((symbolFeaturesLocation, symbolHMMDatesList[hmmDateIdx]))
    createDate = os.listdir(symbolModelFeaturesDate)[2].split("_")[7]  # stupid hack

    # output looks like this: /media/ak/DataOnly/FinDataReal/PRU.L/MODEL_BASED/20170710
    symbolEachModelFeaturesDates = [file.split("_")[5] for file in os.listdir(symbolModelFeaturesDate)]
    # output is a list of dates

    commonDates = list(
        set(symbolEachModelFeaturesDates) & set(symbolLabelsDates))  # this is a list of features and labels dates


for idxKey, commonDate in enumerate(commonLocs.keys()):

    print("create Date:###", createDate)
    print("common Date:-----", commonDate)
    labelsCommonDateFile = ".".join((commonDate, 'csv'))
    labelsCommonFileLoc = "/".join((symbolLabelsLocation, labelsCommonDateFile))
    labelsDf = pd.read_csv(labelsCommonFileLoc)
    comnDateFeatureLocMaster: str = ("/".join((symbolFeaturesLocation, hmmDate)))
    commonDatesFeatureFile = "".join((symbols[1], '_3_states_features_date:_', commonDate, "_now:_", createDate, "_.pickle"))
    FeatureFileLoc = "/".join((comnDateFeatureLocMaster, commonDatesFeatureFile))

    commonLocs[commonDate] = [FeatureFileLoc, labelsCommonFileLoc]
    print(commonLocs[commonDate])

    for idxKey in range(13):
        featuresIdxDirFileLoc = commonLocs[keys[idxKey]][0]
        labelsIdxDirFileLoc = commonLocs[keys[idxKey]][1]

        featuresTupleFile = pickle.load(open(featuresIdxDirFileLoc, "rb"), encoding='latin1')
        labelsDf = pd.read_csv(labelsIdxDirFileLoc)

        dfFeatures = pd.concat([featuresTupleFile[0], featuresTupleFile[1], \
                                    featuresTupleFile[2], featuresTupleFile[3]], axis=1, sort=False).fillna(0)
        # take the labels out
        labels = labelsDf['label_PrMov__window_5__thres_arbitrary__0.1']
        print('preprocessing data...', end='')
        dfXY = pd.concat([dfFeatures, labels], axis=1, sort='False').dropna()
        labelName = str(dfXY.columns[dfXY.columns.str.contains(pat='label')].values[0])
        dfX = dfXY.drop(columns=[labelName])
        print("Shape of dfX..", dfX.shape[0])
        arrX = np.array(dfX)
        X = rescale_01(arrX)  # feature scaling in [0,1]
        X = normalization(rescale_01(arrX))
        print(X.shape, 'done')
        y = dfXY[dfXY.columns[dfXY.columns.str.contains(pat='label')]].iloc[:, 0]
        print("Shape of y..", y.shape)
        Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=.55, random_state=42)
        # compute homogeneous polynomial kernels with degrees 0,1,2,...,10.
        print('computing Homogeneous Polynomial Kernels...', end='')
        KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(4)]
        KLte = [pairwise.homogeneous_polynomial_kernel(Xte, Xtr, degree=d) for d in range(4)]
        from sklearn.metrics.pairwise import rbf_kernel

        gamma_range = np.logspace(-9, 3, 13)
        ker_list = [rbf_kernel(Xtr, gamma=g) for g in gamma_range]
        print('RBF Kernels done')
            ### test set work ###
        KLtest = [pairwise.homogeneous_polynomial_kernel(TestX, Xtr, degree=d) for d in range(4)]

        gamma_range = np.logspace(-9, 3, 13)
        ker_list = [rbf_kernel(Xtr, gamma=g) for g in gamma_range]
        print('training AverageMKL...', end='')
        clf = AverageMKL().fit(KLtr, Ytr)  # a wrapper for averaging kernels
        print('done')
        print(clf.weights)  # print the weights of the combination of base kernels
        print('training EasyMKL...', end='')
        clfEasy = EasyMKL(lam=0.1).fit(KLtr, Ytr)  # combining kernels with the EasyMKL algorithm
        clfRBF = EasyMKL(lam=0.1).fit(ker_list, Ytr)
        # lam is a hyper-parameter in [0,1]
        print('done')
        print(clfEasy.weights)
        print('Linear')
        y_pred = clf.predict(KLte)  # predictions
        y_score = clf.decision_function(KLte)  # rank
        accuracy = accuracy_score(Yte, y_pred)
        roc_auc = roc_auc_score(Yte, y_score)
        print('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))
        print('MKL-Linear')
        y_predTest = clfEasy.predict(KLte)  # predictions
        y_scoreTest = clfEasy.decision_function(KLte)  # rank
        accuracy = accuracy_score(Yte, y_predTest)
        roc_auc = roc_auc_score(Yte, y_scoreTest)
        print('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))
        print('MKL-RBF')
        y_predRBF = clfRBF.predict(KLte)  # predictions
        y_scoreRBF = clfRBF.decision_function(KLte)  # rank
        accuracyRBF = accuracy_score(Yte, y_predRBF)
        roc_aucRBF = roc_auc_score(Yte, y_scoreRBF)
        print('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracyRBF, roc_aucRBF))

