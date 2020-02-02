import os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

np.seterr(divide='ignore', invalid='ignore')
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.model_selection import cross_val_score, cross_val_predict
import pickle as pkl
###
from MKLpy.metrics import pairwise
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc

np.seterr(divide='ignore', invalid='ignore')

from MKLpy.preprocessing import normalization, rescale_01
from sklearn.metrics.pairwise import rbf_kernel as RBF
import os
import pandas as pd
import numpy as np
import pickle as pkl

from MKLpy.metrics.pairwise.misc import homogeneous_polynomial_kernel as HPK_kernel
from MKLpy.metrics import pairwise

import mkl_base as mkl_base
from mkl_base import hardDrivesLoc, dataOnlyDrive, folderList, symbols, \
    dataList, symbolFeaturesLocation, selection

# from aknotebooks.classification.convenience_functions.fit_mkl import featureCreation

'''  Drives / Locations  '''

hardDrivesLoc = '/media/ak/'
dataOnlyDrive = '/media/ak/DataOnly'  # external date only drive
ext_drive_loc = '/media/ak/My Passport/Experiment Data/'
#  input drive
inputDrive = hardDrivesLoc


def featureCreation(idxKey, locDict):
    """ gives out clean features and labels for a given locDict and a idxKey """
    keys = list(locDict.keys())
    featuresIdxDirFileLoc = locDict[keys[idxKey]][0]
    labelsIdxDirFileLoc = locDict[keys[idxKey]][1]
    ''' read the features file'''
    featuresTupleFile = pkl.load(open(featuresIdxDirFileLoc, "rb"), encoding='latin1')
    dfFeatures = pd.concat([featuresTupleFile[0], featuresTupleFile[1], \
                            featuresTupleFile[2], featuresTupleFile[3]], axis=1, sort=False).fillna(0)
    ''' read the labels file'''
    labelsDf = pd.read_csv(labelsIdxDirFileLoc)
    ''' pop the labels out'''
    labels = labelsDf['label_PrMov__window_5__thres_arbitrary__0.1']
    '''dataframe of Features and Labels - X and Y'''
    dfXY = pd.concat([dfFeatures, labels], axis=1, sort='False').dropna()
    labelName = str(dfXY.columns[dfXY.columns.str.contains(pat='label')].values[0])
    ''' drop the labels from the features'''
    dfX = dfXY.drop(columns=[labelName])
    arrX = np.array(dfX)
    ''' feature normalisation'''
    # feature scaling in [0,1] - X = rescale_01(arrX)
    X = normalization(rescale_01(arrX))
    y = dfXY[dfXY.columns[dfXY.columns.str.contains(pat='label')]].iloc[:, 0]
    ''' returns features, labels'''
    return X, y


def forwardDates(idxKey, commonLocs):
    """ return all the forward looking dates for each idxKey we use for training"""
    keys = list(commonLocs.keys())  # this produces a list with 2 keys -first HMM Symbol Date and common Date
    lookAheadKeys = sorted(i for i in keys if i > keys[idxKey])
    return dict((k, commonLocs[k]) for k in lookAheadKeys)


if __name__ == '__main__':

    ''' Inputs '''

    folderIdx = 0

    folderList = [s for s in os.listdir(dataOnlyDrive) if s.startswith('Dat') or s.startswith('Fin')]
    finalLocation = "/".join((dataOnlyDrive, folderList[folderIdx]))  # /media/ak/DataOnly/FinDataReal
    symbols = [s for s in os.listdir(finalLocation) if s.endswith('.L')]  # keep a list of the symbols

    # picking up a specific symbol
    symbolIdx = 6

    # pick one of the symbols
    # symbols[symbolIdx] -->output :PRU.L
    # # do a join to get the location
    print(symbols[symbolIdx])
    symbolLocation = "/".join((finalLocation, symbols[symbolIdx]))
    # # get the features now
    symbolFeaturesLocation = "/".join((symbolLocation, 'MODEL_BASED'))  # where all the HMM output is
    # symbolLocation = "/".join((finalLocation, symbols[symbolIdx]))
    # # # get the features now
    # symbolFeaturesLocation = "/".join(("/".join(("/".join((dataOnlyDrive, folderList[0])), symbols[symbolIdx])),
    #                                    'MODEL_BASED'))  # where all the HMM output is
    # print(symbolFeaturesLocation) # <-- all the HMM model output is here, for each model there is a Date Folder and
    # then OOS files
    locIdx = 1  # '''WorkDrive'''

    selection = os.listdir(inputDrive)[locIdx]

    selectionLoc = os.path.join(inputDrive, selection)  # /media/ak/WorkDrive
    # ''' location of WorkDrive'''
    dataList = [s for s in os.listdir(selectionLoc) if s.startswith('Dat')]
    DataLoc = os.path.join(hardDrivesLoc, selection, dataList[1])
    path = 'MKL_Experiments'
    MKLExpPath = os.path.join(DataLoc, path)
    MKLSymbolPath = os.path.join(MKLExpPath, symbols[symbolIdx])
    # os.makedirs(os.path.join(DataLoc,path)) # run once so comment out afterwards- we can write an if statement later
    ''' Labels and Dates'''

    MKLSymbolKernelsPath = "/".join((MKLSymbolPath, 'Kernels'))

    cleanListKernelInputKeys = pkl.load(open("/".join((MKLSymbolKernelsPath, "cleanKernelsList.pkl",)), "rb"),
                                        encoding='latin1')
    ''' creating the appropriate paths and making the code far more modular'''
    SymbolCommonPaths = mkl_base.open_pickle_file(MKLSymbolPath,
                                                  'LocDictsListCorrect.pkl')  # where the locations of the symbol
    # paths are
    kernelInputPaths = mkl_base.open_pickle_file(MKLSymbolPath,
                                                 'kernelInputsLocations.pkl')  # where the location of the kernel

    symbolHMMDatesList = os.listdir(symbolFeaturesLocation)

    # list of all the MODEL dates we have generated features files for. each #
    # each of these dates in symbolFeaturesDates corresponds to a list of dates
    # (symbolHMMDatesList = '20170829', '20170710', '20170801', ... ]
    # location of labels : /media/ak/DataOnly/FinDataReal/Labels/[Symbol :PRU.L]/NON_DIRECTIONAL

    symbolLabelsLocation = "/".join((finalLocation, 'Labels', symbols[symbolIdx], 'NON_DIRECTIONAL'))

    count_i = 0

    for j in range(0, 5):
        Xdat, ydat = featureCreation(j, SymbolCommonPaths)

        Xtr, Xte, ytr, yte = train_test_split(Xdat, ydat, test_size=.3, random_state=0)
        print(Xtr.shape[0] == ytr.shape[0], Xte.shape[0] == yte.shape[0])
        try:

            ''' do the kernel fitting'''
            KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(4)]
            ''' Compute RBF Kernels'''
            KLte = [pairwise.homogeneous_polynomial_kernel(Xte, Xtr, degree=d) for d in range(4)]
            gamma_range = np.logspace(-9, 3, 13)
            ker_list_tr = [rbf_kernel(Xtr, gamma=g) for g in gamma_range]
            ker_list_te = [rbf_kernel(Xte, gamma=g) for g in gamma_range]
            print('training EasyMKL...for polynomials and RBF', end='')

            ''' fit the classifiers'''
            clf = AverageMKL().fit(KLtr, ytr)  # a wrapper for averaging kernels
            clfEasy = EasyMKL(lam=0.1).fit(KLtr, ytr)  # combining kernels with the EasyMKL algorithm
            clfRBF = EasyMKL(lam=0.1).fit(ker_list_tr, ytr)
            base_learner = SVC(C=100, gamma=10)
            clfSVC = EasyMKL(learner=base_learner)
            clfSVC = clf.fit(KLtr, ytr)
            print('------')
            print('finished training')
        except:
            count_i += 1
            print(count_i)
            print(j, "hin failed here!")

        ''' need to find a smarter way to do the rest'''
        y_pred = clf.predict(KLte)  # predictions
        y_score = clf.decision_function(KLte)  # rank
        accuracy = accuracy_score(yte, y_pred)
        roc_auc = roc_auc_score(yte, y_score)
        print('Average Kernel Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))

        ''' Test Linear'''
        print('SimpleMKL-Linear Testing')
        y_pred_easy = clfEasy.predict(KLte)  # predictions
        y_score_easy = clfEasy.decision_function(KLte)  # rank
        accuracy_easy = accuracy_score(yte, y_pred_easy)
        roc_auc_easy = roc_auc_score(yte, y_score_easy)

        print('EasyMKL Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy_easy, roc_auc_easy))

        ''' Test Linear'''
        print('SVC Base Learner-Linear Testing')
        y_pred_SVC = clfSVC.predict(KLte)  # predictions
        y_score_SVC = clfSVC.decision_function(KLte)  # rank
        accuracy_SVC = accuracy_score(yte, y_pred_SVC)
        roc_auc_SVC = roc_auc_score(yte, y_score_SVC)

        print('MKL with SVC RBF Base Learner Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy_SVC, roc_auc_SVC))





