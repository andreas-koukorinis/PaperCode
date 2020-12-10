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
        self.hmm_dates_list = os.listdir(self.symbol_features_path)  # each folder are the OOS features from each HMM
        self.compute_date = os.listdir(os.path.join( \
            self.symbol_features_path, \
            os.listdir(self.symbol_features_path)[1]))[1].split("_")[7]

    def ticker_features(self, model_date, date):
        # need to make this a lot more flexible with number of states
        if model_date < date:
            file_name = "_".join(
                (self.ticker, '3', 'states', 'features', 'date:', date, 'now:', self.compute_date, '.pickle'))
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

        clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, verbose=1, n_jobs=-1, cv=10)
        clf = clf.fit(self.X_train, self.y_train)
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


def open_pickle_file(path, pickle_file):
    file_loc = os.path.join(path, pickle_file)
    pickle_to_file = pickle.load(open(file_loc, "rb"))
    return pickle_to_file


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)

    # check length
    if len(a_set.intersection(b_set)) > 0:
        return (a_set.intersection(b_set))
    else:
        return ("no common elements")


if __name__ == '__main__':

    #   data_dir: main directory , data_only_drive: the big drive where everything is saved
    # data only dir: main drive that has the

    data_dir = os.getenv('FINANCE_DATA')
    data_only_drive = '/mnt/usb-Seagate_Expansion_Desk_NA8XEHR6-0:0-part2'  # external date only drive

    # this is the central location for all the features/models/predictions
    features_models = os.path.join(data_dir, 'features_models')

    # this is the central location for all the labels
    labels = os.path.join(features_models, 'labels')
    # this is the central location for all the features
    features = os.path.join(features_models, 'features')


    # location to save results
    # this needs to be to data-only drive9
    model_save_loc = os.path.join(data_only_drive, 'Data', 'features_models', 'models')
    # from the main directory select all the symbols that are finishing in .L for FTSE
    symbols_ftse = [s for s in os.listdir(features) if s.endswith('.L')]
    good_symbols = ['LLOY.L']

    # 'CPI.L', 'ULVR.L',  'ECM.L', 'AV.L', 'GKN.L', 'TSCO.L',  'ITV.L',  'BARC.L', 'CPG.L', 'AAL.L', \
    # 'LGEN.L', 'LAND.L', 'VOD.L', 'HSBA.L', 'RSA.L', 'RDSa.L', 'PRU.L', 'III.L', 'REL.L', 'CNA.L', 'SHP.L', 'MKS.L', 'DMGOa.L', 'RR.L', 'DGE.L', 'BATS.L','MAB.L',]
    # 'CPI.L', 'ULVR.L',  'ECM.L', 'AV.L', 'GKN.L', 'TSCO.L',  'ITV.L',  'BARC.L', 'CPG.L', 'AAL.L', \
    # 'LGEN.L', 'LAND.L', 'VOD.L', 'HSBA.L', 'RSA.L', 'RDSa.L', 'PRU.L', 'III.L', 'REL.L', 'CNA.L', 'SHP.L', 'MKS.L']
    #  ['DMGOa.L', 'RR.L', 'DGE.L', 'BATS.L','MAB.L','KGF.L', 'SPT.L', 'AZN.L']
    #1 ['CPG.L','ITV.L','ULVR.L','LGEN.L']
    # this is the central location for all the saved HMM models
    # problems'MKS.L',

    main_path = os.path.join(data_dir, 'features_models')  # main directory

    features_path = os.path.join(main_path, 'features')  # all the features
    labels_path = os.path.join(main_path, 'labels')  # all the labels
    # location to save results
    model_loc = os.path.join(data_only_drive, 'Data', 'features_models', 'models')

    from collections import defaultdict
    all_symbols_d = defaultdict(dict)
    symbol_model_dates = dict()

    for symbol in good_symbols:  # for all symbols
        datacls = DataLoader(path_main=data_dir, ticker=symbol)  # testpcd
        print symbol
        symbol_labels_path = os.path.join(labels_path, symbol, 'NON_DIRECTIONAL')
        symbol_features_path = os.path.join(features_path, symbol, 'MODEL_BASED')
        hmm_models_dates_list = os.listdir(
            symbol_features_path)  # list of all the OOS pre-computed lists of features- each indexed by the date
        for hmm_idx, hmm_date in enumerate(sorted(hmm_models_dates_list)):
            oos_features_date_path = os.path.join(symbol_features_path,
                                                  hmm_models_dates_list[hmm_idx])  # location list of feature files
            oos_dates_list = sorted([oos_date.split("_")[5] for oos_date in
                                     sorted(os.listdir(oos_features_date_path))])  # list of oos features
            oos_labels_features = [(os.path.join(symbol_labels_path, cd  + '.csv'),\
                                    os.path.join(oos_features_date_path, \
                                                 "_".join((symbol, '3', 'states_features_date:', oos_date, 'now:',
                                                           symbol_compute_date, '.pickle')))) \
                                   for oos_date in oos_dates_list]
            all_symbols_d[symbol][hmm_idx] = oos_labels_features

    for k in all_symbols_d.keys():
        print ('doing this Symbol', k) #for each symbol

        fitted_model_symbol_path = os.path.join(data_only_drive, 'Data', 'features_models', 'models', str(k),
                                                'SINGLE_KERNEL')
        for v in all_symbols_d[k].keys():
            print ('second key', v) # and each out of sample that an hmm was fitted
            # each key has a set of no_days which are itemised by a number
            # go through each one and iterate over the components which are really out of sample dates
            for day_no in (all_symbols_d[k].keys()):
                print(day_no)
                print('doing symbol:', k, 'for date:', day_no)

                # this is the location essentially of all the files for that day, i.e all the forward feature files
                # so for each fitted hmm date, there is a list of these files for each symbol
                # this is the list and it contains the full locations so they can be read easily

                oos_file_location = all_symbols_d[k][day_no]
                # if there are no files, just skip
                if len(oos_file_location) == 0:
                    continue
                else:
                    # otherwise go into that location and pull out all the various files you are working with
                    for idx, file_locs in enumerate(oos_file_location):
                        print idx
                        oos_label_date_no = file_locs[0].split("/")[-1].split(".")[0] # strip out the date- will be used later
                        fit_model_sav_loc = os.path.join(fitted_model_symbol_path, oos_label_date_no)
                        symbol_model_dates[k] = oos_label_date_no
                        df_labels = pd.read_csv(file_locs[0], index_col=0)
                        with open(file_locs[1], 'rb') as f:
                            features_tuple = pickle.load(f)

                        features_df = pd.concat([features_tuple[0], features_tuple[1],\
                                                 features_tuple[2], features_tuple[3]], axis=1, sort=False)

                        df_w_market_features = MarketFeatures(df=MarketFeatures( \
                            df=MarketFeatures(
                                df=MarketFeatures(df=df_labels).obv_calc()).chaikin_mf()).ma_spread()).ma_spread_duration()

                        df_concat = pd.concat([features_df, df_w_market_features], axis=1, sort='False').dropna()

                        # ok start putting in the magic
                        # y_duration = np.asanyarray(df_concat['Duration'].shift(window).dropna())
                        # y_price = np.asanyarray(df_concat['TradedPrice'].shift(window).dropna())

                        # drop things we dont need: traded price, duration, traded time, labels etc!
                        label_name = str(df_concat.columns[df_concat.columns.str.contains(pat='label')].values[0])

                        df_final = df_concat.drop(columns=['TradedPrice', 'Duration', 'TradedTime', 'ReturnTradedPrice', \
                                                           'Volume', label_name])
                        # get the labels
                        if df_final.shape[0] < 10: # basically if we have too many fucking nans
                            continue
                        else:
                            X_train = MinMaxScaler().fit_transform(df_final)

                            y_labels = df_concat[df_concat.columns[df_concat.columns.str.contains(pat='label')]].iloc[:, 0]

                    # #  get the label name to use for saving the clf name.oos_date_no
                    # # this can be done a little better as it is re-used in a few places
                            if y_labels.shape[0] - y_labels.value_counts()[0] < 5:
                                print(' the ratio of classes is too low. try another label permutation')
                                continue
                            else:
                                models_cls = FitModels(X_train, y_labels)
                                best_clfs = {'SVC': models_cls.best_svm_clf(kernel_choice="rbf"),
                                             'RF_clf': models_cls.best_random_forest_clf()}

                                seq_clf = "_".join((str(k), str(oos_label_date_no), label_name, "clf", "fitted", ".pickle" ))
                                if not os.path.exists(fit_model_sav_loc):
                                    os.makedirs(fit_model_sav_loc)
                                save_loc = os.path.join(fit_model_sav_loc, seq_clf)
                                pickle.dump(best_clfs, open(save_loc, 'wb'))
                                print('just saved: ', save_loc)

                    # create a hash like dictionary where you can basically
                    # store everything so you can easily access all the models
                                model_spec_hash = {'all_symbol_locations': all_symbols_d, 'model_symbol_dates': symbol_model_dates,
                                                   'labels_config': label_name}

                                with open(os.path.join(model_loc, 'clfs_model_hash.pickle'), 'wb') as handle:
                                    pickle.dump(model_spec_hash, handle, protocol=pickle.HIGHEST_PROTOCOL)

