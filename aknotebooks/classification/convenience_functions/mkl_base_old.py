class CreateElements(object):
    """
    Utility class that creates forward dates from a dictionary and a specific key, and reads and returns features from a
    set of features. The idea is to be able to create these as we go along and not all in one go
    ---------
    attributes:
    idxKey: correspond to the numerical location of a key
    commonLocs: corresponds to a dictionary of Locations- features/labels
    """

    def __init__(self, idxKey, commonLocs):
        self.idx = idxKey
        self.commonLocs = commonLocs

    def forwardDates(self):
        """ return all the forward looking dates for each idxKey we use for training"""
        keys = list(self.commonLocs.keys())  # this produces a list with 2 keys -first HMM Symbol Date and common Date
        lookAheadKeys = sorted(i for i in keys if i > keys[self.idxKey])
        return dict((k, self.commonLocs[k]) for k in lookAheadKeys)

    def featureCreation(self):
        """ gives out clean features and labels for a given location directory and a idxKey """
        keys = list(self.commonLocs.keys())
        featuresIdxDirFileLoc = self.commonLocs[keys[self.idxKey]][0]
        labelsIdxDirFileLoc = self.commonLocs[keys[self.idxKey]][1]
        ''' read the features file'''
        featuresTupleFile = pickle.load(open(featuresIdxDirFileLoc, "rb"), encoding='latin1')
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


'''  Drives / Locations  '''

hardDrivesLoc = '/media/ak/'
dataOnlyDrive = '/media/ak/DataOnly'  # external date only drive
# ext_drive_loc = ('/media/ak/My Passport/Experiment Data/')

data_dir = os.getenv('FINANCE_DATA')  # internal folder with finance data
ext_drive_loc = '/media/ak/My Passport/Experiment Data/'
#  input drive
inputDrive = hardDrivesLoc
''' Inputs '''
folderIdx = 0
folderList = [s for s in os.listdir(dataOnlyDrive) if s.startswith('Dat') or s.startswith('Fin')]
finalLocation = "/".join((dataOnlyDrive, folderList[0]))  # 'FinDataReal' location
finalLocation = "/".join((dataOnlyDrive, folderList[folderIdx]))  # /media/ak/DataOnly/FinDataReal
symbols = [s for s in os.listdir(finalLocation) if s.endswith('.L')]  # keep a list of the symbols

''''picking a symbol'''
symbolIdx = 1  # pick one of the symbols - symbols[symbolIdx] -->output :PRU.L

# do a join to get the location
# picking up a specific symbol
symbolIdx = 1

# pick one of the symbols
# symbols[symbolIdx] -->output :PRU.L
# # do a join to get the location
symbolLocation = "/".join((finalLocation, symbols[symbolIdx]))

# get he features now
symbolFeaturesLocation = "/".join((symbolLocation, 'MODEL_BASED'))

''' 
symbolFeaturesLocation: 
all the HMM model features output is here, for each model there is a Date Folder and then OOS files 
'''
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

symbolHMMDatesList = os.listdir(symbolFeaturesLocation)

# list of all the MODEL dates we have generated features files for. each #
# each of these dates in symbolFeaturesDates corresponds to a list of dates
## (symbolHMMDatesList = '20170829', '20170710', '20170801', ... ]
# (symbolHMMDatesList = '20170829', '20170710', '20170801', ... ]
# location of labels : /media/ak/DataOnly/FinDataReal/Labels/[Symbol :PRU.L]/NON_DIRECTIONAL

symbolLabelsLocation = "/".join((finalLocation, 'Labels', symbols[symbolIdx], 'NON_DIRECTIONAL'))
@@ -65,128 +132,88 @@

symbolLabelsDates = [dateFile.split(".")[0] for dateFile in os.listdir(symbolLabelsLocation)]

'''
            ##now lets go down into each HMM-model date, and pick all the forward futures (out of sample) 
'''

# now lets go down into each HMM-model date, and pick all the forward futures (out of sample)
hmmFeatureLocations = {}  # symbol-hmm-model-date index --> this is the indexation in symbolFeaturesDatesList
commonDatesDict = {}
# this is a struct that will contain for each HMM date, the common labels/features- this should
commonDatesDict = {}  # this is a struct that will contain for each HMM date, the common labels/features- this should
# be used for training and testing
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

    for commonDate in np.sort(commonDates):
        try:
            print("create Date:###", createDate)
            print("common Date:-----", commonDate)
            labelsCommonDateFile = ".".join((commonDate, 'csv'))
            labelsCommonFileLoc = "/".join((symbolLabelsLocation, labelsCommonDateFile))
            labelsDf = pd.read_csv(labelsCommonFileLoc)
            comnDateFeatureLocMaster = ("/".join((symbolFeaturesLocation, hmmDate)))
            commonDatesFeatureFile = "".join(
                (symbols[1], '_3_states_features_date:_', commonDate, "_now:_", createDate, "_.pickle"))
            FeatureFileLoc = "/".join((comnDateFeatureLocMaster, commonDatesFeatureFile))

            commonLocs[commonDate] = [FeatureFileLoc, labelsCommonFileLoc]
            print(commonLocs[commonDate])

        except IndexError: #  catch the error
            continue# pass will basically ignore it

        keys =list(commonLocs.keys())
        #
        # for idxKey in range(4):
        #     try:
        #         ''' get locations of features and labels'''
        #         featuresIdxDirFileLoc = commonLocs[keys[idxKey]][0]
        #         labelsIdxDirFileLoc = commonLocs[keys[idxKey]][1]
        #         ''' start with features construction'''
        #
        #         # featuresTupleFile = pickle.load(open(featuresIdxDirFileLoc,"rb"), encoding='latin1')
        #         # labelsDf=pd.read_csv(labelsIdxDirFileLoc)
        #         # dfFeatures = pd.concat([featuresTupleFile[0], featuresTupleFile[1],\
        #         #                                                  featuresTupleFile[2], featuresTupleFile[3]], axis=1, sort=False).fillna(0)
        #         # ''' take the labels out '''
        #         # labels =labelsDf['label_PrMov__window_5__thres_arbitrary__0.1']
        #         # print ('preprocessing data...', end='')
        #         # dfXY = pd.concat([dfFeatures, labels], axis=1, sort='False').dropna()
        #         # labelName = str(dfXY.columns[dfXY.columns.str.contains(pat='label')].values[0])
        #         # ''' pure features dataframe'''
        #         # dfX = dfXY.drop(columns=[ labelName])
        #         # print("Shape of dfX..", dfX.shape[0])
        #         # arrX = np.array(dfX)
        #         #
        #         # X = normalization(rescale_01(arrX)) #feature scaling in [0,1]
                # print(X.shape, 'done')
                # ''' labels array'''
                # y = dfXY[dfXY.columns[dfXY.columns.str.contains(pat='label')]].iloc[:, 0]
                # print("Shape of y..", y.shape)
                # '''split training set to test set but always on the same day'''
                # Xtr, Xte, Ytr, Yte = train_test_split(X,y, test_size=.55, random_state=42)
                # '''#compute homogeneous polynomial kernels with degrees 0,1,2,...,10.'''
                # print('computing Homogeneous Polynomial Kernels...', end='')
                # KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(4)]
                # KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(4)]
                #
                # gamma_range = np.logspace(-9, 3, 13)
                # ker_list = [rbf_kernel(Xtr, gamma=g) for g in gamma_range]
                # print ('RBF Kernels done')
                # '''
                # start training
                # '''
                #
                # print ('training AverageMKL...', end='')
                # clf = AverageMKL().fit(KLtr,Ytr) #a wrapper for averaging kernels
                # print (' average MKL done')
                # #print(clf.weights) #print the weights of the combination of base kernels
                #
                # '''Combining kernels with the EasyMKL algorithm  '''
                #
                # print ('training EasyMKL for polynomial and RBF Kernels...', end='')
                # clfEasy = EasyMKL(lam=0.1).fit(KLtr, Ytr)
                # clfRBF = EasyMKL(lam=0.1).fit(ker_list, Ytr)
                # print ('MKL Linear and RBF Done')
                # #perhaps store them at some point here
                # #lam is a hyper-parameter in [0,1]
                # #clfs = [clf, clfEasy, clfRBF]
                #
                # #for clfIdx in clfs:
                #    # print (clfEasy.weights)
                #
                #
                #     # print('Average Linear')
                #     # y_pred = clf.predict(KLte)                 #predictions
                #     # y_score = clf.decision_function(KLte)      #rank
                #     # accuracy = accuracy_score(Yte, y_pred)
                #     # roc_auc = roc_auc_score(Yte, y_score)
                #     # print ('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))
                #     # print('MKL-Linear')
                #     # y_predLinearMKL = clfEasy.predict(KLte)                 #predictions
                #     # y_scoreTest = clfEasy.decision_function(KLte)      #rank
                #     # accuracy = accuracy_score(Yte, y_predTest)
                #     # roc_auc = roc_auc_score(Yte, y_scoreTest)
                #     # print ('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))
                #     # print('MKL-RBF')
                #     # y_predRBF = clfRBF.predict(KLte)                 #predictions
                #     # y_scoreRBF = clfRBF.decision_function(KLte)      #rank
                #     # accuracyRBF = accuracy_score(Yte, y_predRBF)
                #     # roc_aucRBF = roc_auc_score(Yte, y_scoreRBF)
                #     # # print ('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracyRBF, roc_aucRBF))
                # except IndexError: #  catch the error
                #     continue# pass will basically ignore it 
symbolEachModelFeaturesDates = {}
HMMModelFeaturesLabelsCommon = {}  # location dictionary with 2 keys: HMM Date and Common Date
commonDates = []

LocDictsList = []


# this symbolFeaturesDatesList[featrsIdx] will give you a date: 20170710 =which contains all the HMM induced feature
def getLocationsList():
    for hmmDateIdx, hmmDate in enumerate(sorted(symbolHMMDatesList)):
        symbolModelFeaturesDate = os.path.join(symbolFeaturesLocation, symbolHMMDatesList[hmmDateIdx])
        create_date = os.listdir(symbolModelFeaturesDate)[0].split("_")[7]  # stupid hack
        # output looks like this: /media/ak/DataOnly/FinDataReal/PRU.L/MODEL_BASED/20170710
        symbolEachModelFeaturesDates[symbolHMMDatesList[hmmDateIdx]] = [file.split("_")[5] for file in
                                                                        os.listdir(symbolModelFeaturesDate)]
        # output is a dictionary where the keys are the HMM models dates and the values a list of dates - for each HMM
        # date we have a list of features

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
                    (symbols[symbolIdx], '_3_states_features_date:_', commonDate, "_now:_", create_date, "_.pickle"))
                FeatureFileLoc = os.path.join(symbolModelFeaturesDate, commonDatesFeatureFile)
                checkDir(FeatureFileLoc)
                checkDir(labelsCommonFileLoc)
                conditions = [os.path.exists(FeatureFileLoc), os.path.exists(labelsCommonFileLoc)]
                print(conditions)
                if all(conditions):
                    print('all good on Date:', commonDate)
                    HMMModelFeaturesLabelsCommon[keyHMMDate, commonDate] = [FeatureFileLoc, labelsCommonFileLoc]
                    pkl.dump(HMMModelFeaturesLabelsCommon,
                             open("/".join((MKLSymbolPath, "LocDictsListCorrect.pkl")), "wb"))

                else:
                    print('problem on date: ', commonDate)
                    continue
                pkl.dump(commonDatesDict,
                         open("/".join((MKLSymbolPath, "CommonLocationsDicts.pkl")), "wb"))

        # LocDictsList.extend(commonDatesDict)
        # LocDictsList.extend(HMMModelFeaturesLabelsCommon)

    return LocDictsList


def mklSymbPath():
    return MKLSymbolPath


def pickleLocsList(loc_dicts_list):
    pkl.dump(loc_dicts_list, open("/".join((MKLSymbolPath, "LocDictsList.pkl")), "wb"))


# dump it all in the same location# #
if __name__ == '__main__':
    # getLocationsList()
    print(symbols)
    print("Picked Symbol:", symbols[symbolIdx])
    print(symbolLocation)
    print(symbolFeaturesLocation)
    checkDir(symbolFeaturesLocation)
    print(selectionLoc)
    print(DataLoc)
    checkDir(DataLoc)
    print(MKLExpPath)
    checkDir(MKLExpPath)
    print(MKLSymbolPath)
    checkDir(MKLSymbolPath)
    getLocationsList()
 68  aknotebooks/classification/convenience_functions/mkl_predict.py 
