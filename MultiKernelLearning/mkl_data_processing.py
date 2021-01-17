import sys

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import pickle
import pandas as pd
import numpy as np
import os
import time
import new_alternate_single_svm as nalsvm
import jsonpickle

# file to do the data processing of clean data so we can fit models quite easily and reduce memory
# just pick symbols I have joint locations
jointLocsSymbols = list(np.unique([f.split("_")[0] for f in os.listdir(nalsvm.jointLocationsPickleFolder)]))

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, hamming_loss


def evaluate_predictions(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    takes true label values and true predictions and returns a dictionary of metrics
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


def storage_location(symbol):
    """

    :param symbol: symbol that we have locations of Data that is clean, i.e matched labels and features
    :return: full absolute path
    """
    storage_location = os.path.join(nalsvm.dataDrive, 'JointLocationsAlternateDataClean', symbol)
    return storage_location


def oos_results_location(symbol):
    oos_results_location = os.path.join(nalsvm.dataDrive, 'JointLocationsAlternateDataClean', 'OOS_Results', symbol)
    if not os.path.exists(oos_results_location):
        os.makedirs(oos_results_location)
    return oos_results_location


def cross_validation_results_location(symbol):
    storage_location = os.path.join(nalsvm.dataDrive, 'JointLocationsAlternateDataClean', 'CV_Results', symbol)
    if not os.path.exists(storage_location):
        os.makedirs(storage_location)
    return storage_location


def model_dates_list(cross_val_location):
    """
    :param cross_val_location: location of cross validation files. this is an output of function cross_validation_results_location
    :return: a sorted list of dates that can refer to when the cross-validation happened
    """
    pkl_files = os.listdir(cross_val_location)
    modelDates = sorted(np.unique([pkl_files[f].split("_")[1] for f, _ in enumerate(pkl_files)]))
    return list(sorted(modelDates))


# results/locations etc
def return_cross_val_symbol_path(symbol):
    """
    returns the path for cross validation results of a symbol for post processing
    if the symbol in the cv_results sub-folder
    input: symbol
    returns: symbol_cross val path
    """
    symbols_available = [f for f in os.listdir(
        '/media/ak/My Passport/Data/FinDataReal/JointLocationsAlternateDataClean/CV_Results/') if str('.L') in f]
    if symbol in symbols_available:
        cross_val_location = cross_validation_results_location(symbol)
        print(cross_val_location)
    else:
        print('symbol not in list')
    return cross_val_location


def pickle_dump_obj_to_filename(destinationPath, symbol, fileName, obj):
    """
    Using this to dump results for a list to file
    :param destinationPath: path where the pickle should be dumped
    :param symbol: Symbol that should accompany the file
    :param fileName: the specific filename you want to use, like OOSResults.pkl
    :param obj: the object you want to pickle
    :return: dumps an obj to file
    """
    pickle_out_filename = os.path.join(destinationPath, "_".join((symbol, fileName)))
    pickle_out = open(pickle_out_filename, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    print('saved', pickle_out_filename)

def jsonpickle_store_obj(obj, filename_location):
    """
    function to encode an object in jsonpickle and store it
    obj: object to be encoded in jsonpickle
    location: where to store it

    """
    obj_str = jsonpickle.encode(obj)
    f = open(filename_location, 'w')
    f.write(obj_str)
    f.close()
    print('encoded and saved in :', filename_location)

if __name__ == '__main__':
    symbol = 'RB.L'
    storage_location = storage_location(symbol)
    if not os.path.exists(storage_location):
        os.makedirs(storage_location)
    good_list_dict = dict()

    if symbol in jointLocsSymbols:
        for alternate_label_idx in range(0, 4):
            print(symbol, ' and labels ', nalsvm.labels_pickle_files[alternate_label_idx])
            data_cls = nalsvm.AlternateLabelFeaturesLoader(path_main=nalsvm.dataDrive,
                                                           symbol=jointLocsSymbols[jointLocsSymbols.index(symbol)],
                                                           alternate_label_idx=alternate_label_idx,
                                                           jointLocationsPickleInput=nalsvm.jointLocationsPickleFolder)
            jointLocationsDictionary = (data_cls.load_pickled_in_filename(data_cls.return_pickled_dict()))
            joint_keys = data_cls.joint_loc_pickle_keys(data_cls.return_pickled_dict())
            nalsvm.logmemoryusage("Before garbage collect")
            nalsvm.gc.collect()  # continue
            for joint_key_idx, joint_key_date in enumerate(joint_keys):
                nalsvm.gc.collect()
                # this is a date - and we will enumerate through the keys
                # getting features and labels
                # nalsvm.logmemoryusage("Before feature creation")
                features, labels = nalsvm.ticker_features_labels(jointLocationsDictionary[joint_keys[joint_key_idx]])
                print('Joint Key Date:', joint_key_date)
                label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
                features_df = nalsvm.hmm_features_df(features)
                nalsvm.gc.collect()
                # features data-frame - this just unbundles the features into a dataframe
                # lets get all the features in order now#
                market_features_df = nalsvm.CreateMarketFeatures(nalsvm.CreateMarketFeatures(
                    nalsvm.CreateMarketFeatures(df=nalsvm.CreateMarketFeatures(
                        df=labels).ma_spread_duration()).ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
                nalsvm.gc.collect()
                # cleaned up prior versions of the code so less memory is used here. dont need all these dataframes!
                df_concat = pd.DataFrame(pd.concat([features_df, market_features_df], axis=1, sort='False').dropna())
                Xtr = df_concat.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                              'Volume', label_name])
                Ytr = df_concat[df_concat[label_name].notna()][df_concat[df_concat[label_name].notna()].columns[
                    df_concat[df_concat[label_name].notna()].columns.str.contains(pat='label')]].iloc[:, 0]
                if Xtr.shape[0] < 3:
                    print(Xtr.shape[0])
                    print('problem')
                    nalsvm.gc.collect()
                else:
                    print('-----------no problem-----------')
                    print(Xtr.shape[0])
                    good_list_dict[joint_key_date] = [Xtr, Ytr]
                    nalsvm.gc.collect()

                pickle_out_filename = os.path.join(storage_location, "_".join(
                    (symbol, nalsvm.labels_pickle_files[alternate_label_idx], 'ProcessedData.pkl')))
                pickle_out = open(pickle_out_filename, 'wb')
                pickle.dump(good_list_dict, pickle_out)
                pickle_out.close()
                print('Data stored:', pickle_out_filename)
