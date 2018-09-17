import pickle
import os
import sys
import pandas as pd
import numpy as np
import sklearn
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import decomposition
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle
# Added version check for recent scikit-learn 0.18 checksok ca
from distutils.version import LooseVersion as Version
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

from sklearn.model_selection import train_test_split
##############----########### Take out what is redundant below###
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

###Huber Norm Models###
sys.path.append('/home/ak/Documents/Research/QFPaper/Huber-Norm-Regularizer')
from huber_svm import HuberSVC
from classification_utilities import off_set, simpleInputs, ensure_dir, symbol_path_
###

###paths###

# Where to save:
#main directory
_pkl_path ='/home/ak/Documents/Data/features_models/'
#graphs directory
_graphs_= os.path.join(_pkl_path, 'graphs')
#models directory
_models_ = os.path.join(_pkl_path, 'models')
#predictions dir
_predictions_ =os.path.join(_pkl_path, 'predictions')
#figures dir
_figures_= ('/home/ak/Documents/Research/QFPaper/paper/figures/')
###labels
labels_ = os.path.join(_pkl_path, 'labels')
###main code###


###scalers###
mms = MinMaxScaler()
stdsc = StandardScaler()
_clfs=[]
class DataLoader:
    def __init__(self, path_, ticker_):
        self.path_ =path_
        self.ticker_ = ticker_
    def ticker_features(self):
        with open(self.path_ + self.ticker_ + '_features.pickle', 'rb') as handle:
            ticker_features_ = pickle.load(handle)
        return ticker_features_

    def ticker_labels(self):
        with open(self.path_ + self.ticker_ + '_labelled_data.pickle', 'rb') as handle:
            ticker_labels_ = pickle.load(handle)
        return ticker_labels_


###main code###
for ticker in symbols_ftse: #ticker = 'VOD' ##load data
    print ticker
    ###the following bit is a bit redundant###
    _load_data= DataLoader(_pkl_path, ticker)
    _data = _load_data.ticker_labels()  # data
    _features = _load_data.ticker_features() # features
    _keys = _data.keys()  # dates

    M = len(_keys) - 1  # number of models- essentially one for each day
    T = 1
    ###array for results###

    _clfs_svm =[]

    all_scores={
        'test-mean': np.empty((M, 5)),
        'test-std':  np.empty((M, 5)),
        'train-mean': np.empty((M, 5)),
        'train-std': np.empty((M, 5)),
        }
# 5 is from the train sizes below
    _fitted_model_results = {
        'clfs': np.empty((M, T)),
        'svm_test_F1': np.empty((M, T)),
        'svm_data_date': np.empty((M, T)),
        'svm_test_recall': np.empty((M, T)),
        'svm_train_recall': np.empty((M, T)),
        'svm_test_accuracy': np.empty((M, T)),
        'svm_train_accuracy': np.empty((M, T)),
    }

    for _idx in xrange(0, M):  # replace this w _xr
        print ticker #ignore
        print ('Doing Day:', _idx) #ignore
        ticker_path_ = os.path.join(_models_ ,ticker)
        #make sure a model/symbol directory exists
        # various types of labels
        y_cc = _data[_keys[_idx]].iloc[:, 6].dropna().astype(int).values  # label_simple_ep_window_5_thresh_0.00-last 4 are NaN
        y_duration = _data[_keys[_idx]].iloc[:, 3].dropna().astype(int).values
        y_price = _data[_keys[_idx]].iloc[:, 1].dropna().astype(int).values  # can be used for regression

        # feature set
        _gamma = _features[_keys[_idx]][2]
        _csi = _features[_keys[_idx]][3]
        _info = _features[_keys[_idx]][1] #keep this out
        _fischer = _features[_keys[_idx]][0]
        short_rolling = _data[_keys[_idx]]['TradedPrice'].rolling(window=5).mean()
        long_rolling = _data[_keys[_idx]]['TradedPrice'].rolling(window=15).mean()
        ma_signal = (long_rolling - short_rolling).fillna(0)

        df_features = pd.concat([_csi, _gamma, _fischer], axis=1)

        df_features['MA_signal'] = \
            _data[_keys[_idx]]['TradedPrice'].rolling(window=15).mean() \
            - _data[_keys[_idx]]['TradedPrice'].rolling(window=5).mean()
        df_features = df_features.dropna()

        X, y = simpleInputs(df_features, y_cc)  # simple inputs just makes sure all the sizes fit


        # # Train a SVM classification model
        param_grid = dict(kernel=["rbf"], C=[1, 5, 10, 25, 50], gamma=[0.0001, 0.001, 0.01])

        X_mms = mms.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_mms, y, test_size=.1, random_state=0)

        # SVC is more expensive so we do a lower number of CV iterations:
        n_jobs=-1
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, verbose=1, n_jobs=-1, cv=5)
        logreg = linear_model.LogisticRegression(C=1e5)
        SGD_clf = SGDClassifier(loss="modified_huber", penalty="elasticnet", n_jobs = -1,learning_rate = 'optimal', n_iter = 2000, alpha = 0.00001, fit_intercept=True)

        #model_fit
        clf.fit(X_train, y_train)  # fitting the first model
        logreg.fit(X_train, y_train) # fitting the second model
        SGD_clf.fit(X_train, y_train) #fitting the third model -modified Huber elastic net
        # fit the model


        clf_name = ticker_path_+'/'+str(_keys[_idx])+'_svm_clf'+ '.pkl' #unique model name
        logreg_name = ticker_path_+'/'+str(_keys[_idx])+'_log_reg'+ '.pkl'
        sgd_clf_name = ticker_path_ + '/' + str(_keys[_idx]) + '_sgd_reg' + '.pkl'


        # _clfs_svm.append(clf) #this is the list of all models-need to save them

        # print('Accuracy: %.2f' % accuracy_score(y_test, y_predict))
        # print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_predict))
        filename = _models_ + str(ticker) + '_' + str(_keys[_idx]) + '_svm_clf' + '.pkl'
        #### saving the model of each day###

        train_sizes = np.linspace(.1, 1.0, 5)

        best_c = clf.best_params_['C']
        best_gamma = clf.best_params_['gamma']
        estimator = SVC(C=best_c, cache_size=200, class_weight='balanced', coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma=best_gamma, kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False)
        estimator.fit(X_train, y_train)
        # save best model/day in a directory
        pickle.dump(estimator, open(clf_name, 'wb'))  # save each model/day in a directory
        pickle.dump(logreg, open(logreg_name, 'wb'))
        pickle.dump(SGD_clf, open(sgd_clf_name, 'wb'))

        ######
        # score curves, each time with 20% data randomly selected as a validation set.

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        all_scores['test-mean'][_idx, :] = np.mean(test_scores, axis=1)
        all_scores['train-mean'][_idx, :] = np.mean(train_scores, axis=1)
        all_scores['train-std'][_idx, :] = np.std(train_scores, axis=1)
        all_scores['test-std'][_idx, :] = np.std(test_scores, axis=1)
        all_scores_file = ticker_path_ + '/' + str(_keys[_idx]) + '_all_scores_single_svm_'
        with open(str(all_scores_file) + '.pkl', 'wb') as w:
            pickle.dump(all_scores, w)
        # ###clf_results
        y_predict = clf.predict(X_test)  # predict on test set
        y_predict_train = clf.predict(X_train)
        _fitted_model_results['svm_test_accuracy'][_idx, :] = accuracy_score(y_test, y_predict)
        _fitted_model_results['svm_test_recall'][_idx, :] = recall_score(y_true=y_test, y_pred=y_predict)
        _fitted_model_results['svm_train_accuracy'][_idx, :] = accuracy_score(y_train, y_predict_train)
        _fitted_model_results['svm_train_recall'][_idx, :] = recall_score(y_true=y_train, y_pred=y_predict_train)
        _fitted_model_results['svm_test_F1'][_idx, :] = f1_score(y_true=y_test, y_pred=y_predict)
        clf_results = ticker_path_ + '/' + str(_keys[_idx]) + '_results_single_svm_'
        with open(str(clf_results) + '.pkl', 'wb') as f:
            pickle.dump(_fitted_model_results, f)












