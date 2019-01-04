import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier #support from multiclass
import time
from sklearn.svm import SVC


class DataLoader(object):
    def __init__(self, path_main, ticker):
        self.main_path = path_main
        self.ticker = ticker

        self.features_labels_path = os.path.join(self.main_path, 'features_models')
        self.features_path = os.path.join(self.features_labels_path, 'features')
        # collection of per symbol non directional labels
        self.labels_path = os.path.join(self.features_labels_path, 'labels', self.ticker, 'NON_DIRECTIONAL')
        self.symbol_features_path = os.path.join(self.features_labels_path, 'features', self.ticker, 'MODEL_BASED')
        # list of all the model -oos hmm feature dates - each folder is a collection of oos feature dates
        self.oos_dates_list = os.listdir(self.symbol_features_path)

    def ticker_features(self, model_date, date):
        # need to make this a lot more flexible with number of states
        if model_date < date:
            file_name = "_".join((self.ticker, '3', 'states', 'features', 'date:', date, 'now:', '20181224', '.pickle'))
            file_loc = os.path.join(self.symbol_features_path, str(model_date), file_name)
            with open(file_loc, 'rb') as handle:
                ticker_features = pickle.load(handle)
        else:
            print('Loading Feature Date which is in-sample. Change your Model Date')
        return ticker_features

    def ticker_labels_csv(self, date):
        file_loc = os.path.join(self.labels_path, str(date) + '.csv')
        ticker_labels = pd.read_csv(file_loc, index_col=0)
        return ticker_labels

    @staticmethod
    def open_pickle_file(path, pickle_file):
        file_loc = os.path.join(path, pickle_file)
        pickle_to_file = pickle.load(open(file_loc, "rb"))
        return pickle_to_file

    @staticmethod
    def get_date_from_file(file_, numb_):
        return os.path.splitext(file_[numb_])[0]


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

    def obv_calc(self):
        # on balance volume indicator
        self.df['SignedVolume'] = self.df['Volume'] * np.sign(self.df['TradedPrice'].diff()).cumsum()
        self.df['SignedVolume'].iat[1] = 0
        self.df['OBV'] = self.df['SignedVolume']  # .cumsum()
        self.df = df.drop(columns=['SignedVolume'])
        return self.df

    def chaikin_mf(self, period=5):
        # Chaikin money flow indicator
        self.df["MF Multiplier"] = (self.df['TradedPrice'] - (self.df['TradedPrice'].expanding(period).min()) \
                                    - (self.df['TradedPrice'].expanding(period).max() \
                                       - self.df['TradedPrice'])) / (
                                           self.df['TradedPrice'].expanding(period).max() - self.df[ \
                                            'TradedPrice'].expanding(period).min())
        self.df["MF Volume"] = self.df['MF Multiplier'] * df['Volume']
        self.df['CMF_' + str(period)] = self.df['MF Volume'].sum() / self.df["Volume"].rolling(period).sum()
        self.df = self.df.drop(columns=['MF Multiplier', 'MF Volume'])
        return self.df


class FitModels(object):

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # # Train a SVM classification model

    def best_kernel_ridge(self, kernel_choice):
        kr_clf = OneVsRestClassifier(GridSearchCV(KernelRidge(kernel=str(kernel_choice)), cv=10,
                                                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                                              "gamma": np.logspace(-2, 2, 5)})).fit(self.X_train,
                                                                                                    self.y_train)

        return kr_clf

    def best_svm_clf(self, kernel_choice):
        param_grid = dict(kernel=[str(kernel_choice)],
                          C=[1, 5, 10, 25, 50, 100],
                          gamma=[0.0001, 0.001, 0.01, 0.02, 0.05, 0.01])

        clf = OneVsRestClassifier(
            GridSearchCV(SVC(class_weight='balanced'), param_grid, verbose=1, n_jobs=-1, cv=10)) \
            .fit(self.X_train, self.y_train)
        return clf

    def best_gradient_boost_clf(self):
        # this needs to be written properly- just a baseline placeholder here!
        GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                        max_depth=4, max_features='sqrt',
                                        min_samples_leaf=15, min_samples_split=10, loss='huber',
                                        random_state=5)

        gb_boost_clf = OneVsRestClassifier(GBR).fit(self.X_train, self.y_train)

        return gb_boost_clf

    def best_MKL_clf(self):
        pass

    def best_knn_clf(self):
        pass

    def best_random_forest_clf(self):
        pass

    def run_cv(self, clf_class, **kwargs):
        # Construct a kfolds object
        kf = KFold(len(self.y_train), n_folds=10, shuffle=True)
        y_pred = self.y_train.copy()

        # Iterate through folds
        for train_index, test_index in kf:
            X_train_local, X_test_local = self.X_train[train_index], self.X_train[test_index]
            y_train_local = self.y_train[train_index]
            # Initialize a classifier with key word arguments
            clf = clf_class(**kwargs)
            clf.fit(self.X_train, self.y_train)
            y_pred[test_index] = clf.predict(X_test_local)
        return y_pred


class PredictModels(FitModels):
    def __init__(self):
        pass


def no_nans(label):
    return np.sum(np.isnan(label))


def remove_last_element(arr):
    return arr[np.arange(arr.size - 1)]

sc = StandardScaler()


if __name__ == '__main__':

    # sort out locations
    # data_dir: main directory , data_only_drive: the big drive where everything is saved

    data_dir = os.getenv('FINANCE_DATA')
    data_only_drive = '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2' #external date only drive

    # from the main directory select all the symbols that are finishing in .L for FTSE
    ftse_symbols = [s for s in os.listdir(data_dir) if s.endswith('.L')]

    # this is the central location for all the features/models/predictions
    features_models = os.path.join(data_dir, 'features_models')

    # this is the central location for all the labels
    labels = os.path.join(features_models, 'labels')
    # this is the central location for all the features
    features = os.path.join(features_models, 'features')
    # this is the central location for all the saved HMM models
    hmm_models = os.path.join(features_models, 'models', 'HMM')
    # location to save results
    model_save_loc = os.path.join(data_only_drive, 'Data','features_models','models')

    # Picked one random symbol - this has to be done over all symbols:

    symbol_features_path = os.path.join(features, ftse_symbols[1], 'MODEL_BASED')
    print(symbol_features_path) #remove

    datacls = DataLoader(path_main=data_dir, ticker=ftse_symbols[1]) #test symbol -create class

    # list of dates that hmm models have been fitted and used out of sample
    model_dates = datacls.oos_dates_list
    #  picking one such hmm model i.e picking an OOS date

    for model_date_idx, model_date in enumerate(model_dates):
        print(model_date)
        t0 = time.time()
        #  path for hmm model features - so concatenates, the specific symbol->features->HMM model(date)->OOS dates
        feature_files_path = os.path.join(symbol_features_path, model_dates[model_date_idx])

        # get all the feature files for each model date
        list_of_feature_files_for_model_date = os.listdir(os.path.join(symbol_features_path, model_dates[model_date_idx]))
        # # get the dates for those feature files- so this returns a list of dates in the format 'YYYY MM DD'
        feature_dt_model_date = [list_of_feature_files_for_model_date[model_date_idx].split("_")[5] for idx, _ in
                                 enumerate(list_of_feature_files_for_model_date)]
        # # pick a feature file
        # print(feature_dt_model_date)  # remove

        for date_idx, date_features in enumerate(list_of_feature_files_for_model_date):
            test_pickle_file = list_of_feature_files_for_model_date[date_idx]
            print('test file:', test_pickle_file)

        # # get all the features in one go

            features_tuple = datacls.open_pickle_file(path=feature_files_path, pickle_file=test_pickle_file)
            # # get labels file
            df = datacls.ticker_labels_csv(date=list_of_feature_files_for_model_date[model_date_idx].split("_")[5])
            df_w_market_features = MarketFeatures( \
                   df=MarketFeatures(df=MarketFeatures(df=df).obv_calc()).chaikin_mf()).ma_spread()

            features_df = pd.concat([features_tuple[0], features_tuple[1], features_tuple[2], features_tuple[3]], axis=1, sort=False)
            # concatenate features from model and market features

            df_concat = pd.concat([features_df, df_w_market_features], axis=1, sort='False').dropna()

print (time.time() -t0)
    #
    # # right order of things
    #

    # from sklearn.svm import SVR
    #
    # np.random.seed(0)
    # svr = GridSearchCV(SVR(kernel='rbf'), cv=5,
    #                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
    #                                "gamma": np.logspace(-2, 2, 5)})
    # svr.fit(X , y)
    #
