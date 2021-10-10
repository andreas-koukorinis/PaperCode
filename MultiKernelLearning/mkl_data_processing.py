import sys

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf/')
import pickle
import pandas as pd
import numpy as np
import os
import time
from fileutils import DataLoader as DataLoader

sys.path.append('/home/ak/Documents/Research/PaperCode/singlekernelclf')
from fileutils import DataLoader as DataLoader
from fileutils import paths
from collections import defaultdict
import psutil
import jsonpickle

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, hamming_loss

alternate_labels_nos = [1, 2, 3, 4, 5, 6, 7]  # we have 7 alternative data types



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

def open_pickle_filepath(pickle_file):
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file


def forward_Dates(list_of_keys, current_date):
    """
    return all the forward looking dates for each idxKey we use for training

    :param list_of_keys: dates i have model dates for out of sample
    :param current_date: current model date
    :return: forward dates for applying the fitted model
    """
    lookAheadKeys = sorted(i for i in list_of_keys if i > current_date)
    return lookAheadKeys


def logmemoryusage(msg):
    # function to log memory usage
    process = psutil.Process(os.getpid())
    print('Memory usage at %s is %smb.' % (msg, process.memory_info().rss / 1000 / 1000))


if __name__ == '__main__':
    # /media/ak/DataOnly/ExperimentCommonLocs/JointLocationsDicts
    mainPath = paths('main')
    jointFeatureLocation = os.path.join(mainPath, "ExperimentCommonLocs/JointLocationsDicts")

    symbols = sorted(os.listdir(paths('symbols_features')))
    print(symbols)  # all symbols
    symbol_idx = 5
    symbol = symbols[symbol_idx]  # to be serialised so read all the symbols
    print(symbol)
    for symbol in symbols:
        symbolData = DataLoader(mainPath, symbol)  # initiate a path where all the data should be

        featuresLabelsLocations = defaultdict(dict)
        for hmm_date_idx, hmm_date in enumerate(sorted(symbolData.hmm_dates_list)):
            hmm_features_date_path = os.path.join(symbolData.symbol_features_path, hmm_date)
            print(hmm_features_date_path)
            for label_idx in alternate_labels_nos:
                for featuresItem in sorted(os.listdir(hmm_features_date_path)):
                    print(os.path.join(hmm_features_date_path, featuresItem))
                    feature_date = featuresItem.split("_")[5]
                    print(feature_date)
                    print("-------------------------")
                    label_file = os.path.join(symbolData.symbol_specific_label_path(label_idx), feature_date + '.csv')
                    if os.path.isfile(label_file):
                        print(label_file)
                        featuresLabelsLocations[hmm_date][feature_date] = [os.path.join(hmm_features_date_path, featuresItem), label_file]
                    else:
                        print('problem#############################')
                pickle_out_filename = os.path.join(jointFeatureLocation, "_".join((symbol, "hmm_date:", str(hmm_date),
                                                                                  "feature_date", str(feature_date),
                                                                                  "label_id:", str(label_idx),
                                                                                  'ProcessedData.pkl')))
                # pickle_out = open(pickle_out_filename, 'wb')
                # pickle.dump(featuresLabelsLocations, pickle_out)
                # pickle_out.close()
                print('Data stored:', pickle_out_filename)
