import pandas as pd
import numpy as np
from yellowbrick.features import Rank1D
import pickle as pkl
from MKLpy.preprocessing import normalization, rescale_01
import os
import numpy as np
np.set_printoptions(precision=2)
from kernelmethods import KernelMatrix
from kernelmethods import PolyKernel, GaussianKernel, LinearKernel
import matplotlib.pyplot as plt
plt.style.use('classic')
import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.model_selection import CVScores
from yellowbrick.features import FeatureImportances
from yellowbrick.contrib.missing import MissingValuesBar
from yellowbrick.target import ClassBalance, FeatureCorrelation
import pandas as pd
from yellowbrick.classifier.confusion_matrix import *
from sklearn.preprocessing import LabelEncoder
# Feature Analysis Imports
from yellowbrick.features import JointPlotVisualizer
from yellowbrick.features import ParallelCoordinates
from yellowbrick.features import PCADecomposition
from yellowbrick.features import Rank1D, Rank2D
from yellowbrick.features import RadViz

'''
file to create all the various plots for each of the symbols for each of the days
'''


### functions ###
def open_pickle_file(path, pickle_file):
    file_loc = os.path.join(path, pickle_file)
    pickle_to_file = pkl.load(open(file_loc, "rb"))
    return pickle_to_file


def featureCreation4Viz(idxKey, locDict):
    ''' gives out clean features and labels for a given locDict and a idxKey
        use idxKey to inser a key from the locDict keys
        so for a particular day based on the key we get features and labels here
        returns features, labels in one data frame
    '''
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

    return dfXY


if __name__ == '__main__':
    symbolsLocs = '/media/ak/My Passport/Experiment Data/MKLExpPath/' #location in the Passport where all the LocDict aligned data is

    for symbolIdx, symbol in enumerate(os.listdir(symbolsLocs)):
        symbolLoc = "".join((symbolsLocs, os.listdir(symbolsLocs)[symbolIdx]))
        try:
            HMMModelFeaturesLabelsCommon = pkl.load(open("/".join((symbolLoc, "LocDictsListCorrect.pkl")), "rb"),encoding='latin1')
            for i in range(1,15):
                dfXY = featureCreation4Viz(i, HMMModelFeaturesLabelsCommon)
                classes = np.array([0., 1.])
                labelName = str(dfXY.columns[dfXY.columns.str.contains(pat='label')].values[0])
                ''' drop the labels from the features'''
                dfX = dfXY.drop(columns=[labelName])
                arrX = np.array(dfX)
                ''' feature normalisation'''
                # feature scaling in [0,1] - X = rescale_01(arrX)
                X = normalization(rescale_01(arrX))
                y = dfXY[dfXY.columns[dfXY.columns.str.contains(pat='label')]].iloc[:, 0]
                ''' returns features, labels'''
                features = list(dfX.columns.values)

                # # Encode the categorical data with one-hot encoding
                # X = pd.get_dummies(dfXY[features])
                #
                # # Convert unique classes (strings) into integers
                # encoder = LabelEncoder()
                # y = encoder.fit_transform(dfXY[labelName])
                # visualizer = PCADecomposition(scale=True, center=False, col=y)
                # visualizer.fit_transform(X, y)
                # visualizer.show()

                # Instantiate the visualizer with the Covariance ranking algorithm
                shapiroVisualiser = Rank1D(features=features, algorithm='shapiro')

                #class balance
                classBalanceTitle = "Class Balance: " + str(symbol) +'for day: '+str(i)
                classBalance = ClassBalance(
                    labels=classes,
                    colors=["teal", "blue"], title=classBalanceTitle
                )
                ClassBalanceFileName = os.path.join('/home/ak/Documents/Research/Papers/figures',
                                                str(symbol.split('.')[0]+'_'+str(i)) + '_ClassBalance.png')
                classBalance.fit(y)
                classBalance.show(outpath=ClassBalanceFileName)
                classBalance.show()

                #feature correlations

                featureCorrelation = FeatureCorrelation(labels=features, color="rebeccapurple")

                featureCorrelation.fit(X, y)
                featureCorrelationName = os.path.join('/home/ak/Documents/Research/Papers/figures',
                                                str(symbol.split('.')[0]) +'_'+str(i)+ '_FeatureCorrelation.png')

                featureCorrelation.show(outpath=featureCorrelationName)  # Finalize and render the visualizer
                featureCorrelation.show()

                # Instantiate the visualizer for RadViz

                RadVizTitle = "RadViz for 21 Features for symbol: " + str(symbol.split('.')[0]+'_'+str(i))
                RadVisualiser = RadViz(classes=classes, features=features, title=RadVizTitle, size=(1080, 720))
                # Create your custom title

                RadVisualiser.fit(X, y)  # Fit the data to the visualizer
                RadVisualiser.transform(X)  # Transform the data

                RadVizFileName = os.path.join('/home/ak/Documents/Research/Papers/figures',
                                                str(symbol.split('.')[0])+'_'+str(i) + '_RadViz.png')
                RadVisualiser.show(outpath=RadVizFileName)  # Finalize and render the visualizer
                RadVisualiser.show()  # Finalize and render the visualizer

        except:
            continue





