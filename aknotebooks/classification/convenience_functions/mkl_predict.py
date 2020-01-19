import MKLpy
from MKLpy.metrics import pairwise
from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD#KOMD is not a MKL algorithm but a simple kernel machine like the SVM
#evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from MKLpy.metrics import pairwise
from MKLpy.preprocessing import normalization, rescale_01
from MKLpy.preprocessing import kernel_normalization
from MKLpy.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics.pairwise import rbf_kernel as RBF
import os
import pandas as pd
import numpy as np
import pickle as pkl
import aknotebooks.classification.convenience_functions.mkl_base as mkl_base
from aknotebooks.classification.convenience_functions.mkl_base import hardDrivesLoc, dataOnlyDrive, folderList, symbols, \
    dataList, symbolFeaturesLocation, selection
from aknotebooks.classification.convenience_functions.fit_mkl import featureCreation
from MKLpy.metrics.pairwise.misc import homogeneous_polynomial_kernel as HPK_kernel

symbolIdx = 2  # pick one of the symbols

# do a join to get the location
# symbolLocation = "/".join((finalLocation, symbols[symbolIdx]))
# # get the features now

selectionLoc = os.path.join(hardDrivesLoc, selection)

# ''' location of WorkDrive'''
# dataList = [s for s in os.listdir(selectionLoc) if s.startswith('Dat')]
# path = 'MKL_Experiments'
MKLExpPath = os.path.join(os.path.join(hardDrivesLoc, selection, dataList[1]), 'MKL_Experiments')
MKLSymbolPath = os.path.join(MKLExpPath, symbols[symbolIdx])
MKLSymbolKernelsPath = "/".join((MKLSymbolPath, 'Kernels'))

SymbolCommonPaths = mkl_base.open_pickle_file(MKLSymbolPath, 'LocDictsListCorrect.pkl')
uniqueTrainingKeys = np.unique([i[1] for i in SymbolCommonPaths])

if __name__ == '__main__':

    for trainingKey, keyDate in enumerate(list(SymbolCommonPaths)):

        print(trainingKey, keyDate[1])
        Xtr, ytr = featureCreation(idxKey=trainingKey,
                                   locDict=SymbolCommonPaths)  # need to refactor this to read from the mkl_base
        if Xtr.shape[0] == ytr.shape[0]:
            print(list(SymbolCommonPaths.keys())[trainingKey][0])
            kernelFileName = "/".join((MKLSymbolKernelsPath, "".join((keyDate[0], "_Kernels.pkl"))))
            loadPickleKernels = pkl.load(open(kernelFileName, "rb"))
            print('doing the kernels bit')
            KLsimple = loadPickleKernels[0]
            KLrbf = loadPickleKernels[1]
            clf = AverageMKL().fit(KLsimple, ytr)  # a wrapper for averaging kernels
            clfEasy = EasyMKL(lam=0.1).fit(KLsimple, ytr)#combining kernels with the EasyMKL algorithm
            clfRBF = EasyMKL(lam=0.1).fit(KLrbf, ytr)
            print('------')
            print('finished MKL Fitting- Now will test Predictions')
            print('Average Kernel Testing')
            y_pred = clf.predict(KLsimple)  # predictions
            y_score = clf.decision_function(KLsimple)  # rank
            accuracy = accuracy_score(ytr, y_pred)
            fprAverage, tprAverage, thresholdsAverage = roc_curve(ytr.ravel(), y_score.ravel())
            roc_auc = auc(fprAverage, tprAverage)
            print('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))
        else:
            print('Shapes dont match.')
            continue
