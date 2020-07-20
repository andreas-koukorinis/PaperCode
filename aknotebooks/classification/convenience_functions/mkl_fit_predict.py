from MKLpy.preprocessing import normalization, rescale_01
from sklearn.metrics.pairwise import rbf_kernel as RBF
from sklearn.metrics import average_precision_score, recall_score
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
###


from MKLpy.metrics import pairwise
from MKLpy.algorithms import AverageMKL, EasyMKL, \
    KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

import psutil
import gc
####

def logmemoryusage(msg):
    process = psutil.Process(os.getpid())
    print('Memory usage at %s is %smb.'%(msg,process.memory_info().rss/1000/1000))

def featureCreation(idxKey, locDict):
    """ gives out clean features and labels for a given locDict and a idxKey """
    keys = list(locDict.keys())
    featuresIdxDirFileLoc = locDict[keys[idxKey]][0]
    labelsIdxDirFileLoc = locDict[keys[idxKey]][1]
    dataDate = keys[idxKey]
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
    return X, y, dataDate



''' 
Paths 
'''
hardDrivesLoc = '/media/ak/'
dataOnlyDrive = '/media/ak/DataOnly'  # external date only drive
ext_drive_loc = '/media/ak/My Passport/Experiment Data/'

#  input drive
inputDrive = hardDrivesLoc


####

average_kernel_results = {
    'tpr': [],
    'fpr': [],
    'auc': [],
    'train_date': [],
    'data_date': [],
    'test_recall': [],
    'train_recall': [],
    'test_accuracy': [],
    'train_accuracy': [],
    'Average precision-recall score': [],
    'thresholds': [],
    'f1_score': [],
}
MKL_results = {
    'tpr': [],
    'fpr': [],
    'auc': [],
    'train_date': [],
    'data_date': [],
    'test_recall': [],
    'train_recall': [],
    'test_accuracy': [],
    'train_accuracy': [],
    'Average precision-recall score': [],
    'thresholds': [],
    'f1_score': [],
    'weights': [],
}
if __name__ == '__main__':
    ''' Inputs '''
    folderIdx = 0
    folderList = [s for s in os.listdir(dataOnlyDrive) if s.startswith('Dat') or s.startswith('Fin')]
    finalLocation = "/".join((dataOnlyDrive, folderList[folderIdx]))
    symbolsList = sorted([s for s in os.listdir(finalLocation) if s.endswith('.L')])  # keep a list of the symbols
    symbolsToRemove = ['AAL.L', 'BLT.L', 'BARC.L', 'AZN.L', 'CNA.L', 'APF.L']
    symbols = [x for x in symbolsList if x not in symbolsToRemove]
    print(symbols)
    # picking up a specific symbol
    for symbolIdx, _ in enumerate(symbols):
        symbol = symbols[symbolIdx]# pick one of the symbols
        print(symbol)
        # symbols[symbolIdx] -->output :PRU.L
        # do a join to get the location
        symbolLocation = "/".join((finalLocation, symbols[symbolIdx]))
        # # get the features now
        symbolFeaturesLocation = "/".join((symbolLocation, 'MODEL_BASED'))  # where all the HMM output is

        # print(symbolFeaturesLocation) # <-- all the HMM model output is here, for each model there is a Date Folder and
        # then OOS files
        locIdx = 1  # '''WorkDrive'''

        selection = os.listdir(inputDrive)[locIdx]

        selectionLoc = os.path.join(inputDrive, selection)
        # ''' location of WorkDrive'''
        dataList = [s for s in os.listdir(selectionLoc) if s.startswith('Dat')]
        DataLoc = os.path.join(hardDrivesLoc, selection, dataList[1])
        path = 'MKL_Experiments'
        MKLExpPath = os.path.join(DataLoc, path)
        MKLSymbolPath = os.path.join(MKLExpPath, symbols[symbolIdx])

        filename = "/".join((MKLSymbolPath, "LocDictsListCorrect.pkl"))
        if os.path.isfile(filename):
            logmemoryusage("Before loading HMMModelFeaturesLabelsCommon")
            HMMModelFeaturesLabelsCommon = pkl.load(open("/".join((MKLSymbolPath, "LocDictsListCorrect.pkl")),
                                                         "rb"), encoding= 'latin1')
            logmemoryusage("After loading HMMModelFeaturesLabelsCommon")
            count_i = 0
            for i in range(1, (len(HMMModelFeaturesLabelsCommon) - 2)):
                logmemoryusage("Before garbage collect")
                gc.collect()
                logmemoryusage("Before feature creation")
                Xtr, ytr, dataDate = featureCreation(i, HMMModelFeaturesLabelsCommon)
                Xte, yte, testDate = featureCreation(i + 1, HMMModelFeaturesLabelsCommon)
                logmemoryusage("After feature creation")
                print('Doing Dates', dataDate)

                if Xtr.shape[0] == ytr.shape[0]:
                    logmemoryusage("Before starting training")
                    print('Shapes Match- starting training ')
                    # polynomial Kernels ##
                    try:
                        KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(4)]
                        KLte = [pairwise.homogeneous_polynomial_kernel(Xte, Xtr, degree=d) for d in range(4)]
                        print('done')
                        # ''' Compute RBF Kernels'''
                        # gamma_range = np.logspace(-9, 3, 13)
                        # ker_list = [rbf_kernel(Xtr, gamma=g) for g in gamma_range]

                        # and train 3 classifiers ###
                        clf = AverageMKL().fit(KLtr, ytr)  # a wrapper for averaging kernels
                        # print(clf.weights)  # print the weights of the combination of base kernels
                        print('training EasyMKL...for polynomials and RBF')
                        clfEasy = EasyMKL(lam=0.1).fit(KLtr, ytr)  # combining kernels with the EasyMKL algorithm
                        # clfRBF = EasyMKL(lam=0.1).fit(ker_list, ytr)
                        print('------')
                        print('finished training')
                    except:
                        count_i += 1
                        print(count_i)
                        print(i, "hin failed here!")

                        continue

                else:
                    print('Shapes dont match.')
                    pass
                print('Average Kernel Testing')
                try:
                    y_pred_te = clf.predict(KLte)  # predictions average kernel
                    y_pred_tr = clf.predict(KLtr)
                    y_score_te = clf.decision_function(KLte)
                    y_score_tr = clf.decision_function(KLtr)
                    fpr_avg, tpr_avg, thresholds_avg = roc_curve(yte.ravel(), y_score_te.ravel())
                    y_predMKL_te = clfEasy.predict(KLte)  # predictions mkl  test
                    y_predMKL_tr = clfEasy.predict(KLtr)  # predictions mkl  train
                    average_kernel_results['fpr'].append(fpr_avg)
                    average_kernel_results['tpr'].append(tpr_avg)
                    average_kernel_results['train_date'].append(testDate)
                    average_kernel_results['data_date'].append(dataDate)
                    average_kernel_results['Average precision-recall score'].append(\
                        average_precision_score(y_pred_tr, y_score_tr))
                    average_kernel_results['thresholds'].append(thresholds_avg)
                    average_kernel_results['test_recall'].append(recall_score(yte, y_pred_te, average='weighted'))
                    average_kernel_results['train_recall'].append(recall_score(ytr, y_pred_tr, average='weighted'))
                    fpr_avg = None
                    tpr_avg = None
                    y_scoreMKL_te = clfEasy.decision_function(KLte)  # predictions
                    y_scoreMKL_tr = clfEasy.decision_function(KLtr)  # rank
                    fprMKL, tprMKL, thresholdsMKL = roc_curve(yte.ravel(), y_scoreMKL_te.ravel())
                    MKL_results['fpr'].append(fprMKL)
                    MKL_results['tpr'].append(tprMKL)
                    MKL_results['f1_score'].append(f1_score(ytr, y_predMKL_tr, average='macro'))
                    MKL_results['thresholds'].append(thresholdsMKL)
                    MKL_results['weights'].append(clfEasy.weights)
                    MKL_results['train_date'].append(testDate)
                    MKL_results['data_date'].append(dataDate)
                    MKL_results['Average precision-recall score'].append(
                        average_precision_score(y_predMKL_tr, y_scoreMKL_tr))
                    MKL_results['test_recall'].append(recall_score(yte, y_predMKL_te, average='weighted'))
                    MKL_results['train_recall'].append(recall_score(ytr, y_predMKL_tr, average='weighted'))
                    fprMKL = None
                    tprMKL = None

                    # accuracy_average = accuracy_score(yte, y_pred_te)
                    average_kernel_results['test_accuracy'].append(accuracy_score(yte, y_pred_te))
                    average_kernel_results['train_accuracy'].append(accuracy_score(ytr, y_pred_tr))
                    # accuracy_mkl = accuracy_score(yte, y_predMKL_te)
                    MKL_results['train_accuracy'].append(accuracy_score(ytr, y_predMKL_tr))
                    MKL_results['test_accuracy'].append(accuracy_score(yte, y_predMKL_te))
                    average_kernel_results['f1_score'].append(f1_score(ytr, y_pred_tr, average='macro'))
                    print('Accuracy of Average Kernel:', average_kernel_results['f1_score'])
                    fileNameAVG = "".join((str(symbol), "_one_day_ahead_average_kernel_results.pkl"))
                    fileNameMKL = "".join((str(symbol), "_one_day_ahead_multiple_kernel_results.pkl"))
                    pkl.dump(MKL_results, open("/".join((MKLSymbolPath, fileNameMKL)), "wb"))
                    pkl.dump(average_kernel_results, open("/".join((MKLSymbolPath, fileNameAVG)), "wb"))
                    print('about to  save:',fileNameAVG )
                    fileNameAVG = None
                    fileNameMKL = None
                except:
                    continue
