from MKLpy.preprocessing import normalization, rescale_01
from sklearn.metrics.pairwise import rbf_kernel as RBF
import os
import pandas as pd
import numpy as np
import pickle as pkl
import aknotebooks.classification.convenience_functions.mkl_base as mkl_base
from MKLpy.metrics.pairwise.misc import homogeneous_polynomial_kernel as HPK_kernel
from aknotebooks.classification.convenience_functions.mkl_base import hardDrivesLoc, dataOnlyDrive, folderList, symbols, dataList, finalLocation


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



if __name__ == '__main__':
    symbolIdx = 1  # pick one of the symbols

    # do a join to get the location
    # symbolLocation = "/".join((finalLocation, symbols[symbolIdx]))
    # # get the features now
    symbolFeaturesLocation = "/".join(
        ("/".join((finalLocation, symbols[symbolIdx])), 'MODEL_BASED'))  # where all the HMM output is

    locIdx = 1  # '''WorkDrive'''

    selection = os.listdir(hardDrivesLoc)[locIdx]

    selectionLoc = os.path.join(hardDrivesLoc, selection)
    # ''' location of WorkDrive'''
    dataList = [s for s in os.listdir(selectionLoc) if s.startswith('Dat')]
    # path = 'MKL_Experiments'
    MKLExpPath = os.path.join(os.path.join(hardDrivesLoc, selection, dataList[1]), 'MKL_Experiments')
    MKLSymbolPath = os.path.join(MKLExpPath, symbols[symbolIdx])
    MKLSymbolKernelsPath = "/".join((MKLSymbolPath, 'Kernels'))
    KernelsLocations = {}
    SymbolCommonPaths = mkl_base.open_pickle_file(MKLSymbolPath, 'LocDictsListCorrect.pkl')
    uniqueTrainingKeys = [i[1] for i in SymbolCommonPaths]

    for trainingKey, keyDate in enumerate(uniqueTrainingKeys):
        print(trainingKey, keyDate)
        Xtr, ytr = featureCreation(idxKey=trainingKey,
                                   locDict=SymbolCommonPaths)  # need to refactor this to read from the mkl_base
        if Xtr.shape[0] == ytr.shape[0]:
            print(list(SymbolCommonPaths.keys())[trainingKey][1])
            kernelFileName = "".join((keyDate, "_Kernels.pkl"))  # refactor this! to incorporate some sort of labelling!
            print('doing the kernels bit')
            KLsimple = [HPK_kernel(Xtr, degree=d) for d in range(1, 11)]
            KLrbf = [RBF(Xtr, gamma=gamma) for gamma in [1., 10, 100.]]
            print('training EasyMKL...for polynomials and RBF', end='')
            Kernels = [KLsimple, KLrbf]
            pkl.dump(Kernels, open("/".join((MKLSymbolKernelsPath, kernelFileName)), "wb"))
            KernelsLocations[trainingKey, keyDate] = "/".join((MKLSymbolKernelsPath, kernelFileName))
            print('------')
            print('finished kernels')
        else:
            print('Shapes dont match.')
            continue
        pkl.dump(KernelsLocations,
                 open("/".join((MKLSymbolPath, "kernelLocations.pkl")), "wb"))
