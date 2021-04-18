from fileutils import DataLoader as DataLoader
from fileutils import paths
from fileutils import new_feature_utils as nfu
from fileutils.new_feature_utils import CreateMarketFeatures
from clfutils import FitModels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, hamming_loss
import os
import multiprocessing
import psutil
import pickle
from collections import defaultdict
import itertools


def open_pickle_filepath(pickle_file):
    pickle_to_file = pickle.load(open(pickle_file, "rb"), encoding='latin1')

    return pickle_to_file


def forwardDates(list_of_keys, current_date):
    """
    return all the forward looking dates for each idxKey we use for training

    :param list_of_keys: dates i have model dates for out of sample
    :param current_date: current model date
    :return: forward dates for applying the fitted model
    """
    lookAheadKeys = sorted(i for i in list_of_keys if i > current_date)
    return lookAheadKeys


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


if __name__ == '__main__':

    # alternate_labels_nos = [1, 2, 3, 4, 5, 6, 7]  # we have 7 alternative data types
    mainPath = paths('main')
    label_idx = 0  # to be serialised


    fittedModelsPath = os.path.join(mainPath, "ExperimentCommonLocs/FittedModels")
    oosPredictionsPath = os.path.join(mainPath, "ExperimentCommonLocs/OOSPredictions")

    symbols = sorted(os.listdir(paths('symbols_features')))
    print(symbols) # all symbols
    symbol_idx = 1
    symbol = symbols[symbol_idx]  # to be serialised so read all the symbols
    print(symbol)

    alternate_labels_nos = [ 1, 2, 3, 4, 5, 6, 7]  # we have 7 alternative data types
    mainPath = paths('main')
    symbolData = DataLoader(mainPath, symbol) # initiate a path where all the data should be
    pickled_models = [f for f in os.listdir(fittedModelsPath) if str(symbol) in f]  # list of all the pickled models
    print(pickled_models)
    for pickled_model in pickled_models:

        print(pickled_model)
        model_date = (pickled_model.split("_")[1])  # load an HMM date model
        model_path = os.path.join(fittedModelsPath, pickled_model)
        print(model_path)
        print(os.path.isfile(model_path))
        best_svc = open_pickle_filepath(model_path)
        print(best_svc.keys())
        best_svc_key_dates = list(best_svc[str(symbol)].keys())
        print(best_svc_key_dates)

    def oos_prediction_function(symbol, label_idx):
        # put it in one function to do simple parallelization


        for pickled_model in pickled_models: # pick a pickled model

            model_date = (pickled_model.split("_")[1])  # load an HMM date model
            model_path = os.path.join(fittedModelsPath, pickled_model)
            print(model_path)
            print(os.path.isfile(model_path))
            best_svc = open_pickle_filepath(model_path)

            for hmm_date_idx, _ in enumerate(symbolData.hmm_dates_list):
                hmm_date = symbolData.hmm_dates_list[hmm_date_idx]  # get all the dates we have essentially an hmm model
                labels_paths = symbolData.hmm_model_feature_corrsp_labels_files(hmm_date,
                                                                                alternate_labels_nos[label_idx])
                # best_svc_key_dates = list(best_svc[str(symbol)].keys())
                # print(best_svc_key_dates)
                # forwardDatesList = forwardDates(list(labels_paths.keys()), best_svc_key_date)

                #
                # for best_svc_key_date in best_svc_key_dates:
                #
                #     # all the various combinations of HMM dates,
                #     # features models.
                #
                #     # all the labels dates that are after the key date that this model was fitted



                    # all the various paths

                    features_paths = symbolData.hmm_model_date_feature_list_filepaths(hmm_date)[1]

                    # for each forward date

                    for forwardDateKey in forwardDatesList:

                        if model_date < forwardDateKey:  # simple check that your model date is not after your forward date!
                            oos_svc_predictions = defaultdict(dict)
    #
    #                         # get your labels
    #
    #                         labels = pd.read_csv(labels_paths[forwardDateKey])
    #                         label_name = str(labels.columns[labels.columns.str.contains(pat='label')].values[0])
    #
    #                         # create features - first HMM and second some Market Features!
    #
    #                         hmm_features = nfu.hmm_features_df(open_pickle_filepath(features_paths[forwardDateKey]))
    #
    #                         if hmm_features.isnull().values.all():
    #                             print('Problem: your HMM features did not compute properly')
    #                         else:
    #
    #                             market_features_df = CreateMarketFeatures(
    #                                 CreateMarketFeatures(
    #                                     CreateMarketFeatures(df=CreateMarketFeatures(df=labels).ma_spread_duration())
    #                                         .ma_spread()).chaikin_mf()).obv_calc()  # market features dataframe
    #
    #                             df_concat = pd.DataFrame(
    #                                 pd.concat([hmm_features, market_features_df], axis=1, sort='False').dropna())
    #
    #                             df = df_concat[df_concat[label_name].notna()]
    #                             df_final = df.drop(
    #                                 columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice',
    #                                          'Volume', label_name])
    #
    #                             y_test = df[df.columns[df.columns.str.contains(pat='label')]].iloc[:, 0]
    #
    #                             try:
    #
    #                                 X_test = MinMaxScaler().fit_transform(df_final)
    #
    #                                 y_pred = best_svc[str(symbol)][model_date]['SVC'].predict(X_test)
    #                                 print(evaluate_predictions(y_test, y_pred))
    #                                 # store the results
    #                                 results_predict_alias = "_".join(
    #                                     (symbol, forwardDateKey, str(alternate_labels_nos[label_idx])))
    #                                 oos_svc_predictions[results_predict_alias][forwardDateKey] = evaluate_predictions(
    #                                     y_test, y_pred)
    #
    #                             except ValueError:
    #                                 print('value error here:****************************************')
    #                                 continue
    #
    #                     else:
    #
    #                         pass
    #
    #                     # store the results
    #
    #                     print('******* Finished and now saving -*-*-*-')
    #
    #                     pickle_out_filename = os.path.join(oosPredictionsPath, "_".join(
    #                         (symbol, str(alternate_labels_nos[label_idx]), forwardDateKey, 'OOS_results_dict.pkl')))
    #                     pickle_out = open(pickle_out_filename, 'wb')
    #                     pickle.dump(oos_svc_predictions, pickle_out)
    #                     pickle_out.close()
    #                     print('saved', pickle_out_filename)
    #
    #

    #
    # with multiprocessing.Pool(processes=4) as process_pool:
    #     process_pool.starmap(oos_prediction_function, itertools.product(symbols, alternate_labels_nos))
