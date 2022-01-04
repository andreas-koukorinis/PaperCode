
import pandas as pd
import numpy as np
import sklearn.svm as svm
import pickle
import os

from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
standard_scaler = StandardScaler()
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel as RBF
import pickle as pkl
###
from yellowbrick.style import set_palette

#evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.features import Rank1D
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.model_selection import CVScores
from yellowbrick.features import FeatureImportances
from yellowbrick.contrib.missing import MissingValuesBar
from yellowbrick.target import ClassBalance, FeatureCorrelation

from yellowbrick.classifier.confusion_matrix import *

from sklearn.preprocessing import LabelEncoder
# Feature Analysis Imports
from yellowbrick.features import JointPlotVisualizer
from yellowbrick.features import ParallelCoordinates
from yellowbrick.features import PCADecomposition
from yellowbrick.features import Rank1D, Rank2D
from yellowbrick.features import RadViz
from sklearn.cluster import KMeans
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.style import set_palette
from yellowbrick.features.manifold import Manifold
### matplotlib stuff ##

from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl
import matplotlib.pyplot as plt
styleFiles = os.listdir(mpl.get_configdir())
styleFileIdx = 2
print(styleFiles[styleFileIdx])
plt.style.use(os.path.join(mpl.get_configdir(), styleFiles[styleFileIdx]))

'''
file to create all the various plots for each of the symbols for each of the days
'''


### functions ###
def open_pickle_file(path, pickle_file):
    file_loc = os.path.join(path, pickle_file)
    pickle_to_file = pkl.load(open(file_loc, "rb"), encoding='iso-8859-1')
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
    # locations ##
    figuresDestination ='/home/ak/Documents/Research/Papers/figures/'
    hardDriveLoc = '/media/ak/'
    dataOnlyFolder = '/media/ak/DataOnly'  # external date only drive
    symbolFeaturesDirectories  = os.path.join(dataOnlyFolder, 'SymbolFeatureDirectories')
    ExperimentCommonLocs = os.path.join(dataOnlyFolder,'ExperimentCommonLocs')

    labelsLocations = {'One': os.path.join(ExperimentCommonLocs, 'LabelsAlternateOne'),
                       'Two': os.path.join(ExperimentCommonLocs, 'LabelsAlternateTwo'),
                       'Three': os.path.join(ExperimentCommonLocs, 'LabelsAlternateThree'),
                       'Four': os.path.join(ExperimentCommonLocs, 'LabelsAlternateFour'),
                       'File':os.path.join(ExperimentCommonLocs, 'LabelsAlternateFive'),
                       'Six': os.path.join(ExperimentCommonLocs, 'LabelsAlternateSix')}

    # symbols

    symbols = sorted(os.listdir(symbolFeaturesDirectories))

    # pick a symbol

    symbolIdx = 18
    symbolSpecificFeaturesDirectors = os.path.join(symbolFeaturesDirectories, symbols[symbolIdx])
    print('you chose symbol:', symbols[symbolIdx])
    symbolFeaturesLocation = "/".join((symbolSpecificFeaturesDirectors, 'MODEL_BASED'))  # where all the HMM output is
    print('your features are here-in pickle format:',symbolFeaturesLocation)
    # and a label Idx
    labelsIdx = 'Four'
    print('you are picking Label', labelsIdx)
    LabelsFilesPath = os.path.join(labelsLocations['Two'], symbols[symbolIdx])
    LabelsFiles = os.listdir(LabelsFilesPath)

    dates = os.listdir(symbolFeaturesLocation)

    for dateIdx in [5, 10, 15]:
        dateSpecificDirectory = os.path.join(symbolFeaturesLocation, dates[dateIdx])
        dateSpecificFiles = sorted(os.listdir(dateSpecificDirectory))

        for idx in range (0, 14):

            specificFile = os.path.join(dateSpecificDirectory, dateSpecificFiles[idx])

            featuresTupleFile  = open_pickle_file(dateSpecificDirectory, dateSpecificFiles[idx])

            label_date = specificFile.split("_")[6]

            labelDateFile = os.path.join(LabelsFilesPath, str(label_date) +'.csv')

            if os.path.isfile(labelDateFile):
                df = pd.read_csv(labelDateFile)
            else:
                print('not a valid dates file')

            labelName = str(df.columns[df.columns.str.contains(pat='label')].values[0])
            labels = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]

            dfFeatures = pd.concat([featuresTupleFile[0], featuresTupleFile[1], \
                                    featuresTupleFile[2], featuresTupleFile[3]], axis=1, sort=False).fillna(0)


            features= list(dfFeatures.columns.values)
            X = pd.get_dummies(dfFeatures)
            # Convert unique classes (strings) into integers
            encoder = LabelEncoder()
            y = encoder.fit_transform(labels.fillna(0))

            classes= np.unique(labels.fillna(0)) # need this for later

            plt.style.use('seaborn')
            # Instantiate the visualizer with the Covariance ranking algorithm
            my_title = ''

            #
            plt.figure()
            visualizerRank1D = Rank1D(algorithm='shapiro', color=["cadetblue"], title ='   ')

            visualizerRank1D.fit(X, y)
            visualizerRank1D.transform(X)
            plt.xticks(fontsize=13 )
            plt.yticks(fontsize=12 )
            plt.tight_layout()
            locationFileNameRank1D = \
                os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx])+'_idx_'+str(idx)+'_date_'+str(dateIdx)+'_label_'+str(labelName)+'_shapiro.png')
            visualizerRank1D.show(outpath=locationFileNameRank1D)
            plt.show()


            # Instantiate the visualizer
            plt.figure()
            visualizerJPV = JointPlotVisualizer(columns=["lambda_weight", "lambda_sigma"], title = ' ')

            visualizerJPV.fit_transform(X, y)        # Fit and transform the data
            # Finalize and render the figure
            plt.ylabel('lambda_sigma',fontsize=14 )
            plt.xlabel('lambda_weight',fontsize=14 )
            locationFileNameJPV = os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx])+'_idx_'+
                                            str(idx)+
                                            'date'+str(dateIdx)+'_label'+str(labelName)+'_jointplotViz.png')
            visualizerJPV.show(outpath=locationFileNameJPV)
            plt.show()

            # # Instantiate the visualizer with the Covariance ranking algorithm

            set_palette('sns_dark')
            plt.figure()
            visualizerR2D = Rank2D(features=features, algorithm='pearson', title = '  ')

            visualizerR2D.fit(X, y)                # Fit the data to the visualizer
            visualizerR2D.transform(X)             # Transform the data
            plt.xticks(fontsize=12 )
            plt.yticks(fontsize=12 )
            locationFileNameR2D = os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx])+'_idx_'+
                                            str(idx)+'_label'+str(labelName)+'_date_'+str(dateIdx)+'_pearsonCorrel.png')
            visualizerR2D.show(outpath=locationFileNameR2D)
            plt.show()


            my_title = " "
            plt.figure()
            visualizerManifold = Manifold(
                manifold="tsne", target="discrete", colors=["red", "blue"], title= my_title
            )
            visualizerManifold.fit_transform(X, y)      # Fit the data to the visualizer
            locationFileNameManifold = os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx]) \
                                            +'_idx_'+str(idx)+'_date_'+ str(dateIdx) \
                                            +'_label'+str(labelsIdx)\
                                            +'tsne.png')

            visualizerManifold.show(outpath=locationFileNameManifold)
            plt.show()

            my_title = "Random Forest Feature Importance: "+str(symbols[symbolIdx])
            plt.figure()
            model = RandomForestClassifier(n_estimators=10)
            colors = ["lightpink", "pink", "hotpink", "crimson", "orchid"]
            vizRFC = FeatureImportances(model, colors=colors, title=' ')
            plt.xlabel('features',fontsize=18 )
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 12)
            locationFileNameRFC = os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx]) \
                                           +'_label_'+str(labelsIdx)+ '_idx_'+str(idx)+str('Date')+str(dateIdx)+'_RandForImp.png')
            vizRFC.fit(X, y)
            vizRFC.show(outpath=locationFileNameRFC)
            plt.show()

            plt.figure()

            # Instantiate the visualizer
            visualizerFC = FeatureCorrelation(labels=features, color="rebeccapurple", title = ' ')

            visualizerFC.fit(X, y)
            locationFileNameFC = os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx])+'_idx_'+str(idx) \
                                            +'_label_'+str(labelName)+'_date_'+str(dateIdx)+'_label_'+str(labelsIdx)+'_FeatureCorrelation_w_depn_var.png')
            plt.xlabel('',fontsize=11 )
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 12)
            visualizerFC.show(outpath=locationFileNameFC)
            plt.show()
            #
            # # Instantiate the visualizer

            set_palette('yellowbrick')
            plt.figure()
            classes = np.array([ 0,  1.])
            plt.xticks(fontsize=9)
            visualizerRadViz = RadViz(classes=classes, features= features, title = ' ')
            visualizerRadViz.fit(X, y)           # Fit the data to the visualizer
            visualizerRadViz.transform(X)        # Transform the data
            locationFileNameRVZ = os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx]) \
                                            +'_idx_'+str(idx)+'_label_'+str(labelsIdx)+'_date_'+str(dateIdx)+'_radviz.png')
            visualizerRadViz.show(outpath=locationFileNameRVZ)
            plt.show()

            ## MDS


        # Instantiate the clustering model and visualizer
            model = KMeans(6)
            plt.figure()
            plt.xlabel('features',fontsize=12 )
            plt.ylabel('features',fontsize=12 )

            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 12)
            visualizerID = InterclusterDistance(model)
            visualizerID.fit(X) # Fit the data to the visualizer

            locationFileNameID = os.path.join('/home/ak/Documents/Research/Papers/figures',str(symbols[symbolIdx])+'_idx_'+str(idx)+'_KMeans_MDS.png')
            visualizerID.show(outpath=locationFileNameID) # Finalize and render the figure
            plt.show()



