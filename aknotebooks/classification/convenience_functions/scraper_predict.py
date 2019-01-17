from hsmm_core.data_utils import DataLoader, TradingHours
from hsmm_core.feature_spaces import hmm_features
from hsmm_core.hsmm_runner import HmmCalibration
import time
from hsmm_core.consts import InitialisationMethod
from hsmm_core.data_utils import TradingHours, DataLoader
from hsmm_core.labelling import DataLabellingSimple
from hsmm_core.consts import ThresholdMethod, LabellingChoice
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
sc = StandardScaler()
import os
import pickle
import fnmatch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

##useful function


def fwd_dates(_dates_list, _key_date):
    # returns a list of dates that are forward from the key_date
    fwd_dates_list = [i for i in _dates_list if i > _key_date]
    return fwd_dates_list


def remove_nans(features_tuple, labels, idx=1):
    # not the cleanest but useful
    # function to clean up nans as I seem to use it a lot, so better to have one function
    # combines the features and labels and removes rows with nans across so we dont lose the ordering
    # returns features and labels
    features_df = pd.concat([features_tuple[0], features_tuple[1], features_tuple[2], \
                             features_tuple[3]], axis=1, sort=False)
    labels_only = labels.drop(columns=['ReturnTradedPrice', 'Duration', 'states', 'TradedTime',
                                       'TradedPrice'], axis=1)
    df_concat = pd.concat([features_df, labels_only.iloc[:, 0:idx]], axis=1, sort='False')
    # only using 1st set of labels- but we can re-write this a bit
    df_x_nan = df_concat.dropna()  # dropping all nans
    label_column_loc_ = df_x_nan.shape[1] - 1  # location of labels column in the clean df
    labels_ = df_x_nan.iloc[:, label_column_loc_:label_column_loc_ + 1]  # keep pure labels
    features_ = df_x_nan.drop(df_x_nan.columns[label_column_loc_], axis=1)  # keeping the features only
    return features_, labels_


def prec_recall_report(y_true, y_predict):
    # function to ge the sci-kit learn classification metrics into a pretty DF for csv!
    report = pd.DataFrame(list(precision_recall_fscore_support(y_true, y_predict)),
                          index=['Precision', 'Recall', 'F1-score', 'Support']).T
    # Now add the 'Avg/Total' row
    report.loc['Avg/Total', :] = precision_recall_fscore_support(y_true, y_predict, average='weighted')
    report.loc['Avg/Total', 'Support'] = report['Support'].sum()
    return report

class MarketFeatures(object):
    # a class to be expanded that uses features for base case -market based only-indicators/features
    """"Requires:
    a dataframe that has TradedPrice And Volume columns
    symbol - A stock symbol on which to form a strategy on.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average.
    """

    def __init__(self, df):
        #         self.ticker = ticker
        self.df = df

    def load_data(self):
        pass

    def ma_spread(self, short_window=5, long_window=20):
        # function that produces the MA spread, which can be used on its own or as an input for MACD
        short_rolling_px = self.df['TradedPrice'].rolling(window=short_window).mean()
        long_rolling_px = self.df['TradedPrice'].rolling(window=long_window).mean()
        px_name = "_".join(('px_indx', str(short_window), str(long_window)))
        self.df[px_name] = long_rolling_px - short_rolling_px
        return self.df

    def ma_spread_duration(self, short_window=5, long_window=20):
        # function that produces the MA spread, which can be used on its own or as an input for MACD
        short_rolling_px = self.df['Duration'].rolling(window=short_window).mean()
        long_rolling_px = self.df['Duration'].rolling(window=long_window).mean()
        dur_name = "_".join(('dur_indx', str(short_window), str(long_window)))
        self.df[dur_name] = long_rolling_px - short_rolling_px
        return self.df

    def obv_calc(self):
        # on balance volume indicator
        self.df['SignedVolume'] = self.df['Volume'] * np.sign(self.df['TradedPrice'].diff()).cumsum()
        self.df['SignedVolume'].iat[1] = 0
        self.df['OBV'] = self.df['SignedVolume']  # .cumsum()
        self.df = self.df.drop(columns=['SignedVolume'])
        return self.df

    def chaikin_mf(self, period=5):
        # Chaikin money flow indicator
        self.df["MF Multiplier"] = (self.df['TradedPrice'] - (self.df['TradedPrice'].expanding(period).min()) \
                                    - (self.df['TradedPrice'].expanding(period).max() \
                                       - self.df['TradedPrice'])) / (
                                           self.df['TradedPrice'].expanding(period).max() - self.df[ \
                                       'TradedPrice'].expanding(period).min())
        self.df["MF Volume"] = self.df['MF Multiplier'] * self.df['Volume']
        self.df['CMF_' + str(period)] = self.df['MF Volume'].sum() / self.df["Volume"].rolling(period).sum()
        self.df = self.df.drop(columns=['MF Multiplier', 'MF Volume'])
        return self.df


if __name__ == '__main__':

    # locations

    data_dir = os.getenv('FINANCE_DATA')  # main directory referenced in all the code
    data_only_drive = '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2'  # external date only drive

    # this is the central location for all the features/models/predictions
    features_models = os.path.join(data_dir,
                                   'features_models')
    # main path where all the sub-directories are (features, models, labels)

    # this is the central location for all the labels
    labels = os.path.join(features_models, 'labels')  # label subdirectory
    # this is the central location for all the features #feature subdirectory
    features = os.path.join(features_models, 'features')

    # location to save results
    model_save_loc = os.path.join(data_only_drive, 'Data', 'features_models',
                                  'models')  # location where all the symbols are
    # from the main directory select all the symbols that are finishing in .L for FTSE
    symbols_ftse = [s for s in os.listdir(features) if s.endswith('.L')]
    main_path = os.path.join(data_dir, 'features_models')  # main directory
    metrics_loc = os.path.join(data_only_drive, 'Data','features_models','metrics')

    features_path = os.path.join(main_path, 'features')  # all the features - same as above -redundant
    labels_path = os.path.join(main_path, 'labels')  # all the labels

    # same as above- new target directory, where all the models and output is saved

    model_paths = os.path.join(data_only_drive, 'Data', 'features_models',
                               'models')

    # using lambda to make a small function that just takes in the symbol
    # and produces the relevant path of all fitted single kernel models

    def symbol_fitted_models_path(symbol): return os.path.join(model_paths, symbol, 'SINGLE_KERNEL')


    # provides a fitted list of above path

    def symbol_list_fitted_dates(symbol): return sorted(os.listdir(symbol_fitted_models_path(symbol)))


    # fitted model sub-directory- the fitted model is stored in this sub-directory as a pickle

    def symbol_fitted_model_date_loc(file_path, model_date_no):
        return os.path.join(file_path, str(symbol_list_fitted_dates(symbol)[model_date_no]))


    def symbol_model_date_loc(model_date_path):
        return os.path.join(model_date_path, os.listdir(model_date_path)[0])


    def model_pickle_to_svc(model_pickle):

        pickle_to_file = pickle.load(open(model_pickle, "rb"))
        best_estimator = pickle_to_file['SVC'].best_estimator_

        return best_estimator


    # test case ##

    # symbols to use as a starting point
    good_symbols = [
        'RDSa.L', 'PRU.L', 'III.L', 'REL.L', 'CNA.L', 'SHP.L', 'MKS.L',
        'CPI.L', 'ULVR.L', 'ECM.L', 'AV.L', 'GKN.L', 'TSCO.L', 'ITV.L',
        'BARC.L', 'CPG.L', 'AAL.L', 'LGEN.L', 'LAND.L', 'VOD.L', 'HSBA.L',
        'RSA.L', 'DMGOa.L', 'RR.L', 'DGE.L', 'BATS.L', 'MAB.L',
        'KGF.L', 'SPT.L', 'AZN.L'
    ]

    symbol = good_symbols[1]  # picking PRU as an example

    # test symbol path, which essentially produces the path where all the fitted models are.
    # '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2/Data/features_models/models/SPT.L/SINGLE_KERNEL'

    symbol_labels_path = os.path.join(labels_path, symbol, 'NON_DIRECTIONAL')

    symbol_features_path = os.path.join(features_path, symbol, 'MODEL_BASED')

    test_path = symbol_fitted_models_path(symbol=symbol)

    # now lets take all the model directories and locations##
    model_directories = [symbol_fitted_model_date_loc(test_path, idx) for idx, date in enumerate(os.listdir(test_path))]
    models_locations = [os.path.join(model_directories[dir_idx], os.listdir(model_directories[dir_idx])[0]) for
                        dir_idx, model_dir in enumerate(model_directories)]

    for model_idx, model_loc in enumerate(models_locations):
        model_date = model_loc.split("/")[8]
        model_pickle = model_loc

        print model_pickle
        pickle_to_file = pickle.load(open(model_pickle, "rb"))

        best_estimator = pickle_to_file['SVC'].best_estimator_

        print('Your symbol is:', symbol, 'and the model date is:' ,model_date)

        # get all the dates of the labels from the labels path - this may be a bit redundant in the end

        labels_dates = sorted([os.listdir(symbol_labels_path)[idx].split(".")[0]
                               for idx, _ in enumerate(os.listdir(symbol_labels_path))])

        # this is the location of the out of sample features

        oos_features_path = os.path.join(symbol_features_path, model_date)
        oos_dates_list = sorted([oos_date.split("_")[5] for oos_date in
                                 sorted(os.listdir(oos_features_path))])  # list of oos features

        # keep only the fwd dates i.e the oos dates

        fwd_dates = fwd_dates(_dates_list=oos_dates_list, _key_date=model_date)

        # set up the dictionary for metrics #
        M = len(fwd_dates)
        T = 1
        T_2 = 4

        fitted_models_results = {
            'accuracy': np.empty((M,T)),
            'recall': np.empty((M,T)),
            'F1-score': np.empty((M,T)),
            'precision_recall_fscore_support': np.empty((M, T_2))
        }

        for fwd_idx, fwd_date in enumerate(fwd_dates):

            features_tuple = pickle.load(open(os.path.join(oos_features_path, os.listdir(oos_features_path)[fwd_idx]), "rb"))

            market_data_oos = pd.read_csv(os.path.join(labels_path, symbol, 'NON_DIRECTIONAL', fwd_date + '.csv'), index_col=0)
            features_df = pd.concat([features_tuple[0], features_tuple[1],
                                     features_tuple[2], features_tuple[3]], axis=1, sort=False)
            df_w_market_features = MarketFeatures(df=MarketFeatures(\
                df=MarketFeatures(
                    df=MarketFeatures(df=market_data_oos).obv_calc()).chaikin_mf()).ma_spread()).ma_spread_duration()

            df_concat = pd.concat([features_df, df_w_market_features], axis=1, sort='False').dropna()

            # ok start putting in the magic
            # y_duration = np.asanyarray(df_concat['Duration'].shift(window).dropna())
            # y_price = np.asanyarray(df_concat['TradedPrice'].shift(window).dropna())

            # drop things we dont need: traded price, duration, traded time, labels etc!
            label_name = str(df_concat.columns[df_concat.columns.str.contains(pat='label')].values[0])

            df_final = df_concat.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                               'Volume', label_name])

            X = MinMaxScaler().fit_transform(df_final)

            y_labels = df_concat[df_concat.columns[df_concat.columns.str.contains(pat='label')]].iloc[:, 0]
            y_predict = best_estimator.predict(X)

            results_loc = str(os.path.join(metrics_loc, "_".join((symbol,model_date,"results_metrics.pickle"))))
            fitted_models_results['accuracy'][fwd_idx, :] = accuracy_score(y_labels, y_predict)
            fitted_models_results['recall'][fwd_idx, :] = recall_score(y_true=y_labels, y_pred=y_predict)
            fitted_models_results['F1-score'][fwd_idx, :] =f1_score(y_true= y_labels, y_pred=y_predict)
            fitted_models_results['precision_recall_fscore_support'][fwd_idx, :] = precision_recall_fscore_support(y_true=  y_labels, y_pred=y_predict, average='micro')

            with open(results_loc, 'wb') as f:
                pickle.dump(fitted_models_results, f)







