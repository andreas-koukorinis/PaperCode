import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import BDay
from hsmm_core.hmm import hmm_engine
from hsmm_core.observation_models import ExpIndMixDiracGauss
from hsmm_core.feature_spaces import hmm_features
from hsmm_core.hmm import hmm_calibration
from hsmm_core.data_utils import load_data, TradingHours
from hsmm_core.labelling import DataLabellingSimple
from hsmm_core.consts import ThresholdMethod, LabellingChoice
import pickle
from hsmm_core.consts import InitialisationMethod
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
sc = StandardScaler()

###-functions-###

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
                                       'TradedPrice', 'ticker'], axis=1)
    df_concat = pd.concat([features_df, labels_only.iloc[:, 0:idx]], axis=1, sort='False')
    # only using 1st set of labels- but we can re-write this a bit
    df_x_nan = df_concat.dropna()  # dropping all nans
    label_column_loc_ = df_x_nan.shape[1] - 1  # location of labels column in the clean df
    labels_ = df_x_nan.iloc[:, label_column_loc_:label_column_loc_ + 1]  # keep pure labels
    features_ = df_x_nan.drop(df_x_nan.columns[label_column_loc_], axis=1)  # keeping the features only

    return features_, labels_ #return features and labels in the X,y order that scikit takes the input


def prec_recall_report(y_true, y_predict):
    # function to ge the sci-kit learn classification metrics into a pretty DF for csv!
    report = pd.DataFrame(list(precision_recall_fscore_support(y_true, y_predict)),
                          index=['Precision', 'Recall', 'F1-score', 'Support']).T
    # Now add the 'Avg/Total' row
    report.loc['Avg/Total', :] = precision_recall_fscore_support(y_true, y_predict, average='weighted')
    report.loc['Avg/Total', 'Support'] = report['Support'].sum()
    return report

##### classes####

#####OOP#####
class DataLoader(object):
    def __init__(self, path_, ticker):
        self.main_path = path_
        self.ticker = ticker
        self.labels_path = os.path.join(self.main_path, 'labels')
        self.features_path = os.path.join(self.main_path, 'features')
        self.ticker_labels_path = os.path.join(self.labels_path, self.ticker)
        self.ticker_features_path = os.path.join(self.features_path, self.ticker)

    def ticker_features(self, date):
        file_loc = os.path.join(self.ticker_features_path, str(date) + '.pickle')
        with open(file_loc, 'rb') as handle:
            ticker_features = pickle.load(handle)
        return ticker_features

    def ticker_labels_pickle(self, date):
        file_loc = os.path.join(self.ticker_labels_path, str(date) + '.pickle')
        with open(file_loc, 'rb') as handle:
            ticker_labels = pickle.load(handle)
        return ticker_labels

    def ticker_labels_csv(self, date):
        file_loc = os.path.join(self.ticker_labels_path, str(date) + '.csv')
        ticker_labels = pd.read_csv(file_loc)
        return ticker_labels

    @staticmethod
    def open_pickle_file(path, pickle_file):
        file_loc = os.path.join(path, pickle_file)
        pickle_to_file = pickle.load(open(file_loc, "rb"))
        return pickle_to_file

    @staticmethod
    def get_date_from_file(file_, numb_):
        return os.path.splitext(file_[numb_])[0]


class PriceIndicators(object):
    # a class to be expanded that uses features for base case -price only-indicators
    """"Requires:
    symbol - A stock symbol on which to form a strategy on.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""

    def __init__(self, symbol, labels_df):
        self.symbol = symbol
        self.labels = labels_df

    def MACD(self, short_window=5, long_window=20):
        short_rolling_px = self.labels['TradedPrice'].rolling(window=short_window).mean()
        long_rolling_px = self.labels['TradedPrice'].rolling(window=long_window).mean()
        px_indx = long_rolling_px - short_rolling_px
        return px_indx


class FitModels(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # # Train a SVM classification model

    def ridge_clf(self, cv_folds=5):
        model_ridge_clf = RidgeClassifierCV(alphas=np.arange(0.1, 1000, 0.1), \
                                            cv=KFold(cv_folds), normalize=True).fit(self.X_train,
                                                                                    self.y_train.values.ravel())
        # check if class_weight should be used as 'balanced'

        return model_ridge_clf

    def svm_clf(self, kernel_choice):
        param_grid = dict(kernel=[str(kernel_choice)],
                          C=[1, 5, 10, 25, 50, 100],
                          gamma=[0.0001, 0.001, 0.01, 0.02, 0.05, 0.01])
        svc = SVC(class_weight='balanced')
        clf = GridSearchCV(svc, param_grid)
        clf.fit(self.X_train, np.asanyarray(self.y_train).reshape(self.y_train.shape[0]))

        return clf

    def gradient_boost_clf(self, learning_rate=0.25):
        # this needs to be written properly- but this is somewhat optimised#
        GBR = GradientBoostingClassifier(n_estimators=3000, learning_rate=learning_rate,
                                         max_depth=4, max_features='sqrt',
                                         min_samples_leaf=15, min_samples_split=10)

        gb_boost_clf = GBR.fit(self.X_train, self.y_train)

        return gb_boost_clf

    def gp_clf(self):
        kernel = 1.0 * RBF([1.0])  # isotropic
        gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(self.X_train, self.y_train)
        # hyper-parameters are optimised by default
        return gpc_rbf_isotropic

    def random_forest_clf(self, no_est=100):
        rfc = RandomForestClassifier(n_estimators=no_est, max_depth=4, n_jobs=-1, warm_start=True)
        rfc.fit(X_train, y_train)

        return rfc

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


if __name__ == '__main__' :
    #  setting locations
    ticker = 'test_SYNT_2states' #testing a new synthetic ticker

    data_dir = os.getenv('FINANCE_DATA') #main directory
    features_path = '/home/ak/Data/features_models/features/' #where features are saved
    labels_path = '/home/ak/Data/features_models/labels' #where labels are saved
    ticker_labels_path = os.path.join(labels_path, ticker + '/NON_DIRECTIONAL')

    if not os.path.exists(os.path.join(data_dir, ticker)):
        os.makedirs(os.path.join(data_dir, ticker))

    if not os.path.exists(ticker_labels_path):
        os.makedirs(ticker_labels_path)

    labels_list = os.listdir(ticker_labels_path)

    ####paths####
    main_path = '/home/ak/Data/features_models/'

    models_path = os.path.join(main_path, 'models')
    ticker_models_path = os.path.join(models_path, ticker)

    # hmm_models_path = os.path.join(models_path,'hmm_models') #only if we store the hmm models
    if not os.path.exists(ticker_models_path):
        os.makedirs(ticker_models_path)

    print "making paths-done-delete"

    #  Parameters -setting up the HMM etc
    no_states = 2
    sigmas = [0.05, 0.002]  # fast and slow
    # Duration is measured in seconds for now (to be revised). lambda units are seconds^{-1}
    # so here we consider

    lambdas = [1. / 35., 1. / 10.]
    weights = [0.1, 0.6]

    obs_model = ExpIndMixDiracGauss(no_states)
    obs_model.set_up_initials(priors={'sigmas': sigmas, 'lambdas': lambdas, 'weights': weights})

    hmm_ = hmm_engine(obs_model, no_states)

    # set up some priors
    tpm = np.array([[0.4, 0.6], [0.7, 0.3]])
    pi = np.array([0.4, 0.6])
    hmm_.set_up_initials(priors={'tpm': tpm, 'pi': pi})

    no_dates = 30  # <-- this is the number of days you want
    start_date = pd.datetime(2017, 6, 1)
    dummy_dates = [start_date + BDay(i) for i in range(no_dates)]

    no_points = 5000

    rng = np.random.RandomState(1234)
    trd_hours_filter = TradingHours.all_trading_day

    # silly hack, add 1 millisecond so that the initial timestamp is printed with milliseconds and does not
    # break the parsing of Timestamps when loading

    morning_start = dt.time(8, 0, 0, 1)

    initial_price = 100

    for dd in dummy_dates:
        random_states = hmm_.sample_states(rng=rng, length=no_points)
        observation_points = obs_model.sample_data(no_points, rng=rng, state=random_states)
        # The first duration is always zero
        observation_points[0, 0] = 0.

        file_path = os.path.join(data_dir, ticker)
        file_name = '.'.join([dd.strftime('%Y%m%d'), 'csv'])

        data_to_save = pd.DataFrame({'states': random_states,
                                     'Duration': observation_points[:, 0],
                                     'ReturnTradedPrice': observation_points[:, 1],
                                     })
        data_to_save['TradedTime'] = pd.Series()

        # Now calculate the Traded prices and traded times in reverse order as to what would happen
        # with real data.
        # data_to_save.loc[0, 'TradedTime'] = dt.datetime.combine(dd.date(), morning_start)
        data_to_save['TradedTime'] = data_to_save['Duration'].cumsum().apply(lambda dur:
                                                                             (dt.datetime.combine(dd.date(),
                                                                                                  morning_start) + \
                                                                              dt.timedelta(seconds=dur)).time())

        data_to_save['TradedPrice'] = initial_price * (1. + data_to_save['ReturnTradedPrice']).cumprod()
        data_to_save.to_csv(os.path.join(file_path, file_name), index=False)

    print "ok-produced data" # can remove this a bit later

    init_params = {
        "obs_model_params": {
            'obs_model_name': 'ExpIndMixDiracGauss',
            'em_init_method': InitialisationMethod.cluster

        },
        "hidden_model_params": {
            'no_hidden_states': no_states,
            'pi': pi,
            'tpm': tpm,
            'em_init_method': InitialisationMethod.uniform
        },
        "update_tag": 'tpsml'
    }

    data = load_data(ticker, which_trading_hours=TradingHours.all_trading_day) #prob dont need this

    hmm_calibration_engine = hmm_calibration(no_parallel_procs=None,
                                             init_params=init_params)

    trained_hmms = hmm_calibration_engine.hmm_fit_func(ticker, data, trd_hours_filter,
                                                       force_recalc=False)

    for date, date_hmm in trained_hmms.iteritems():
        feature_engine = hmm_features(date_hmm)
        features = feature_engine.generate_features(data[date])

    # Create Labels ###

    window = 25
    threshold = 0.1

    labelling_method_params = [{

        'labelling_method': LabellingChoice.price_move_in_window,
        'rolling_window': window,
        # Uncomment below if you want to check a price move only above a certain level
        'updown_threshold': threshold,  # this is multiplied by 100
        'threshold_method': ThresholdMethod.arbitrary,
    }]

    for label_init in labelling_method_params:
        print label_init
        labeller = DataLabellingSimple(label_init)
        labeller.label_training_data(data)

    # clf fitting##
    for date, date_hmm in trained_hmms.iteritems():
        feature_engine = hmm_features(date_hmm)
        features_load = feature_engine.generate_features(data[date])
        labels_load = pd.read_csv(os.path.join(ticker_labels_path,str(date)+'.csv'))
        features, labels_clean = remove_nans(features_load, labels_load)
        x_std = sc.fit_transform(features.values.astype(np.float))  # fit & transform the features
        X_train, X_test, y_train, y_test = train_test_split(
            x_std, labels_clean, test_size=0.01, random_state=1, stratify=labels_clean)  # probably can get rid of this
        models_cls = FitModels(X_train, y_train)
        best_clfs = {'SVC': models_cls.svm_clf(kernel_choice="rbf"),
                      'RIDGE_clf': models_cls.ridge_clf(),
                     'GBOOST': models_cls.gradient_boost_clf(),
                      'GP_clf': models_cls.gp_clf()
                     # 'RF_clf': models_cls.random_forest_clf(),
                     }
        # This is sequence for the name of the best classifiers.
        seq_clf = "_".join(("synt_model",  str(date), labels_clean.columns.values[0], "clfs", ".pickle"))
        print("saving the classifiers:", seq_clf)
        pickle.dump(best_clfs, open(os.path.join(ticker_models_path, seq_clf), 'wb'))



