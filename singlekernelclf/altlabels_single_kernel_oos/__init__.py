import sys
sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import fileutils as fileutils
import gc
import psutil
import pickle
import fnmatch
import os
import time
import clfutils
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, hamming_loss
import new_alternate_single_svm as nalsvm


def feature_labels_fit(features, labels):
    label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
    features_df = nalsvm.hmm_features_df(
        features)  # features data-frame - this just unbundles the features into a dataframe
    # lets get all the features in order now#
    if features_df.isnull().values.all():
        print('HMM Features Problematic')  # do i need a counter here?
        market_features_df = nalsvm.CreateMarketFeatures(
            nalsvm.CreateMarketFeatures(
                nalsvm.CreateMarketFeatures(df=nalsvm.CreateMarketFeatures(df=labels).ma_spread_duration())
                    .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
        df_concat = market_features_df.dropna()
        df = df_concat[df_concat[label_name].notna()]
        df_final = df.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                    'Volume', label_name])
    else:
        market_features_df = nalsvm.CreateMarketFeatures(
            nalsvm.CreateMarketFeatures(
                nalsvm.CreateMarketFeatures(df=nalsvm.CreateMarketFeatures(df=labels).ma_spread_duration())
                    .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
        df_concat = pd.DataFrame(pd.concat([features_df, market_features_df], axis=1, sort='False').dropna())
        df = df_concat[df_concat[label_name].notna()]
        df_final = df.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                    'Volume', label_name])

    if df_final.shape[0] < 10:
        print(' the ratio of classes is too low. try another label permutation')
    else:
        X_ = MinMaxScaler().fit_transform(df_final)
        y_labels = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]

        return X_, y_labels


def hmm_features_df(features):
    features_df = nalsvm.hmm_features_df(
        features)  # features data-frame - this just unbundles the features into a dataframe
    return features_df


def logmemoryusage(msg):
    # function to log memory usage
    process = psutil.Process(os.getpid())
    print('Memory usage at %s is %smb.' % (msg, process.memory_info().rss / 1000 / 1000))


def unpickle_csv(pickled_csv):
    with open(pickled_csv, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    return p


def ticker_features_labels(file_joint_locations):
    # input a joint location file that contains both features and labels and returns one
    labels = pd.read_csv(file_joint_locations[1])
    features = unpickle_csv(file_joint_locations[0])
    return features, labels


def hmm_features_df(features_tuple):
    return pd.concat([features_tuple[0], features_tuple[1], \
                      features_tuple[2], features_tuple[3]], axis=1, sort=False)


def evaluate_predictions(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='weighted')
    precision_macro = precision_score(y_true, y_preds, average='macro')
    precision_micro = precision_score(y_true, y_preds, average='micro')
    recall = recall_score(y_true, y_preds, average='weighted')
    f1_weighted = f1_score(y_true, y_preds, average='weighted')
    f1_macro = f1_score(y_true, y_preds, average='macro')
    f1_micro = f1_score(y_true, y_preds, average='micro')
    hamming_loss_value = hamming_loss(y_true, y_preds)

    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1- weighted": round(f1_weighted, 2),
                   "f1- micro": round(f1_micro, 2),
                   "f1- macro": round(f1_macro, 2),
                   "Hamming Loss": round(hamming_loss_value, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score weighted: {f1_weighted:.2f}")
    print(f"F1 score macro: {f1_macro:.2f}")
    print(f"F1 score micro: {f1_micro:.2f}")
    print(f"Hamming Loss Value: {hamming_loss_value:.2f}")

    return metric_dict


########################################################################################################################
# paths
dataDrive = '/media/ak/My Passport/Data/FinDataReal/'  # also labels location folder
jointLocationsPickleFolder = os.path.join(dataDrive, 'JointLocationsDicts')
extPath = '/media/ak/My Passport/ExperimentData'
featuresPath = "/".join((extPath, 'features'))  # path with features
experimentPath = os.path.join(extPath,'AlternateLabelExperimentPath' )

# Labels
labels_location_folder = fileutils.data_path  # this is the folder where all the labels are saved

labels_pickle_files = sorted([s for s in os.listdir(labels_location_folder) if ('LabelsAlternate') in s if
                              not ('.pkl') in s])  # these are all the dicts that we have alternate labels for.
# labels_pickle_files: these are all the dicts that we have alternate labels for.

symbols = [f for f in [s for s in os.listdir(labels_location_folder) if '.L' in s if '_Features' not in s] if
           ('.L_A' or '_Features') not in f]  # from all
forwardDates = [f for f in os.listdir(dataDrive) if 'ForwardDates' in f]
svcModels = [g for g in os.listdir(dataDrive) if '_SingleKernelSVC' in g]

if __name__ == "__main__":
    symbol = 'CPI.L'
    # this is a list of all the models that have been fitted, format is Symbol_label_type_single_kernel_svc
    # like this: Symbol_LabelsAlternateOne_SingleKernelSVC.pkl'
    symbolSVCModels = [g for g in svcModels if str(symbol) in g]

    for alternate_label_idx in range (0,4):

        print(alternate_label_idx)  # pick a label by index
        # look into the JointLocationsDicts folder and list all the symbols for which we have joint locations
        # this is needed to make sure you pick a symbol that has both model and data/labels
        # output is a list of symbols
        jointLocsSymbols = list(np.unique([f.split("_")[0] for f in os.listdir(jointLocationsPickleFolder)]))

        # return index of symbol from above list. i.e fit the symbol you
        # chose by ticker and get index back
        symbol_idx = jointLocsSymbols.index(symbol)  # jointLocsSymbols[symbol_idx]--> this returns the symbol.

        # instance of the class with the necessary functions

        data_cls = nalsvm.AlternateLabelFeaturesLoader(path_main=dataDrive, symbol=symbol,
                                                       alternate_label_idx=alternate_label_idx,
                                                       jointLocationsPickleInput=jointLocationsPickleFolder)

        # return a dictionary that has a key a date which is the HMM feature date and the labels
        # the values are the location of such feature file and the labels file
        jointLocationsDictionary = (data_cls.load_pickled_in_filename(data_cls.return_pickled_dict()))

        joint_keys = data_cls.joint_loc_pickle_keys(data_cls.return_pickled_dict())
        # basically the keys of the above dictionary

        print(labels_pickle_files[alternate_label_idx])  # which label I am using - do i really need this?
        logmemoryusage("Before garbage collect")
        gc.collect()  # continue

        # ------------- SVC Model Location and Alternate Labels -------------

        # the bit below returns the symbol _alternate label type_ svc model for all the choices made above
        symbolSVCModelLocationIndexedLabel = \
            [f for f in symbolSVCModels if str(labels_pickle_files[alternate_label_idx]) in f][0]
        # join with the path to get the complete Location, via symbol+ SVC + Model Location
        symbolSVCModelLocation = os.path.join(dataDrive, symbolSVCModelLocationIndexedLabel)
        print(symbolSVCModelLocation)  # just checking you got the right location
        # may need to check I am using the correct label
        alternate_label = symbolSVCModelLocation.split("_")[1]

        # ------------- MODELS -------------
        # unpickle the models - returns a dictionary
        models = unpickle_csv(symbolSVCModelLocation)
        # get all the dates I have models for- returns a list
        modelDates = list(models[list(models.keys())[0]].keys())
        # pick the first date
        for modelDateIdx,_ in enumerate(modelDates): # <-- this is fit-date in sample that corresponds to the labels date too
            print('-------------------------------------------Doing Model date:------>',modelDates[modelDateIdx])
            print('-------------------------------------------For Symbol:------>', symbol)

            # select the first model date model - this returns an scikit object

            svc = models[str(symbol)][modelDates[modelDateIdx]]['SVC']  # model object

            # start measuring time for profiling
            start = time.time()
            # check that your symbol is in the list of symbols
            if symbol in jointLocsSymbols:
                print('ok to go')  # un-necessary logging
                print(symbol, ' and labels ', labels_pickle_files[alternate_label_idx])
                # get the correct features and labels
                features, labels = ticker_features_labels(jointLocationsDictionary[modelDates[modelDateIdx]])
                # take out features and labels properly
                X_fit, y_fit = feature_labels_fit(features, labels)
                # get fitted model object
                try:
                    fitted_model = svc.fit(X_fit, y_fit)
                    # now get a list of all the out of sample dates
                    symbolForwardDates = data_cls.forwardDates(joint_keys, modelDates[modelDateIdx])
                    oos_svc_predictions = defaultdict(dict)
                    # alias to store the data : symbol, joint Date, Label Used
                    results_predict_alias = "_".join((symbol, modelDates[modelDateIdx], alternate_label))

                    for forward_date_idx, forward_date in enumerate(symbolForwardDates):
                        features_oos, labels_oos = ticker_features_labels(
                            jointLocationsDictionary[symbolForwardDates[forward_date_idx]])
                        if hmm_features_df(features_oos).isnull().values.all():
                            print('Problem')
                        else:
                            X_true, y_true = feature_labels_fit(features_oos, labels_oos)
                            try:
                                y_pred = fitted_model.predict(X_true)
                                print(evaluate_predictions(y_true, y_pred))
                                # store the results
                                oos_svc_predictions[results_predict_alias][forward_date] = evaluate_predictions(y_true, y_pred)
                            except ValueError:
                                print('value error here:', forward_date)
                                continue
                except ValueError:

                    continue


                    # store the results

                print('******* Finished and now saving -*-*-*-')

                pickle_out_filename = os.path.join(experimentPath, "_".join(
                    (symbol, labels_pickle_files[alternate_label_idx],modelDates[modelDateIdx], 'OOS_results_dict.pkl')))
                pickle_out = open(pickle_out_filename, 'wb')
                pickle.dump(oos_svc_predictions, pickle_out)
                pickle_out.close()
                print('saved', pickle_out_filename)


                end = time.time()
                print(f'it took {end - start} seconds!')
