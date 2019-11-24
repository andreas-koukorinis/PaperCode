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

##picking up a specific feature
symbolIdx = 1 #pick one of the symbols
#symbols[symbolIdx] -->output :PRU.L
print(symbols[symbolIdx])
# do a join to get the location
symbolLocation = "/".join((finalLocation,symbols[symbolIdx]))

# get he features now
symbolFeaturesLocation = "/".join((symbolLocation,'MODEL_BASED')) # where all the HMM output is
print('This will be deleted, but I am picking this symobl and reading from this folder:')
print(symbolFeaturesLocation) # <-- all the HMM model output is here, for each model there is a Date Folder and then OOS files


symbolFeaturesDatesList = os.listdir(symbolFeaturesLocation)
# list of all the MODEL dates we have generated features files for. each #
# each of these dates in symbolFeaturesDates corresponds to a list of dates
# (symbolFeaturesDatesList = '20170829', '20170710', '20170801', ... ]
# now lets move to labels ...pick the location of the labels and make a list of all the available labels #

#location of labels : /media/ak/DataOnly/FinDataReal/Labels/[Symbol :PRU.L]/NON_DIRECTIONAL

symbolLabelsLocation = "/".join((finalLocation, 'Labels',symbols[symbolIdx],'NON_DIRECTIONAL'))

#list of all the label dates

symbolLabelsDates =[dateFile.split(".")[0] for dateFile in os.listdir(symbolLabelsLocation)]

# Output symbolLabelsDates --> ['20170704', '20180226', '20180208',...] all we are doing is going for this

##now lets go down into each HMM-model date, and pick all the forward futures (out of sample)
featrsIdx = 3 #symbol-hmm-model-date index

#this symbolFeaturesDatesList[featrsIdx] will give you a date: 20170710 =which contains all the HMM induced featureb

symbolOneFeaturesDate = "/".join((symbolFeaturesLocation, symbolFeaturesDatesList[featrsIdx]))
# output looks like this: /media/ak/DataOnly/FinDataReal/PRU.L/MODEL_BASED/20170710

symbolEachModelFeaturesDates=[file.split("_")[5] for file in os.listdir(symbolOneFeaturesDate )]

print("you are picking this HMM model date:",symbolFeaturesDatesList[featrsIdx])

# method to find the list of dates (intersection) that we have both HMM-model-features and labels
# symbolEachModelFeaturesDates # set of OutOfSample Dates produced by each HMM Model
# symbolLabelsDates # set of all the Labels
# we want to match OOS Dates with Labels
# for these dates we have features and we have labels- so we can fit and predict! this produces a list of [date1, date 2, ...]
commonDates =list(set(symbolEachModelFeaturesDates ) & set(symbolLabelsDates))
#Now we need to reconstruct the dates for the fitting
commonIdx =0 #common dates index
# the cache of the file includes the date it was saved, the first date is the Common Date and the next date is the "creation date"
createDate = os.listdir(symbolOneFeaturesDate)[2].split("_")[7] #stupid hack
## now we reconstruct the file but with arbitrary new date ##
commonDatesFeatureDateFile = \
"".join((symbols[1],"_3_states_features_date:_",commonDates[commonIdx], \
         "_now:_",createDate,"_.pickle"))

## now lets load up the pickle and the labels ##
testIdx= 3
featuresIdxDirFileLoc = "/".join((symbolFeaturesLocation, symbolFeaturesDatesList[testIdx] \
                               ,  commonDatesFeatureDateFile))
featuresTupleFile = pickle.load(open(featuresIdxDirFileLoc,"rb"), encoding='iso-8859-1')

dfFeatures = pd.concat([featuresTupleFile[0], featuresTupleFile[1],\
                                                 featuresTupleFile[2], featuresTupleFile[3]], axis=1, sort=False).fillna(0)

labelsCommonDateFile = ".".join((commonDates[commonIdx],'csv'))

labelsCommonFileLoc = "/".join((symbolLabelsLocation, labelsCommonDateFile))
labelsDf=pd.read_csv(labelsCommonFileLoc)
#take the labels out
labels =labelsDf['label_PrMov__window_5__thres_arbitrary__0.1']

labelsShift = labels.isna().sum() # going to use this for "allignment of features and labels"
### put labels and features together before you dropnans ###

dfXY = pd.concat([dfFeatures, labels], axis=1, sort='False').dropna()
labelName = str(dfXY.columns[dfXY.columns.str.contains(pat='label')].values[0])
## get working now on features and labels for MKL ##
dfX = dfXY.drop(columns=[ labelName])
print("Shape of dfX..",dfX.shape[0])
dfX.columns.values

X =StandardScaler().fit_transform(dfX)
y = dfXY[dfXY.columns[dfXY.columns.str.contains(pat='label')]].iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#preprocess data
print ('preprocessing data...', end='')
X = normalization(rescale_01(np.array(dfX)))
Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=.55, random_state=42)
print(X.shape, 'done')

#compute homogeneous polynomial kernels with degrees 0,1,2,...,10.
print('computing Homogeneous Polynomial Kernels...', end='')
KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(4)]
KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(4)]
gamma_range = np.logspace(-9, 3, 13)
ker_list = [rbf_kernel(Xtr, gamma=g) for g in gamma_range]
print ('done')
# MKL algorithms #
print ('training AverageMKL...', end='')
clf = AverageMKL().fit(KLtr,Ytr)  # a wrapper for averaging kernels
print ('done')
print(clf.weights)  # print the weights of the combination of base kernels
K_average = clf.ker_matrix  # the combined kernel matrix
clfRBF = clf = AverageMKL().fit(ker_list,Ytr) #a wrapper for averaging kernels
K_average_rbf = clfRBF.ker_matrix

# training MKL #
print ('training EasyMKL...', end='')
clf = EasyMKL(lam=0.1).fit(KLtr,Ytr)#combining kernels with the EasyMKL algorithm
clfEasyRBF = EasyMKL(lam=0.1).fit(ker_list,Ytr)
#lam is a hyper-parameter in [0,1]
print ('done')
print (clf.weights)