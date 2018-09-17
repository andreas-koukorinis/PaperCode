import pickle
import os
import fnmatch
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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import pickle
# Added version check for recent scikit-learn 0.18 checksok ca
from distutils.version import LooseVersion as Version
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split


# Where to save the figures
_pkl_path =('/home/ak/Documents/features_test/')
ticker='C.N' #enter symbol you want - this will be made for variou
_graphs_path= '/home/ak/Documents/features_test/graphs/'
_models_path = '/home/ak/Documents/features_test/models/'
_predictions_path ='/home/ak/Documents/features_test/predictions/'


def off_set(array_1, array_2):
    off_set= len(array_1)-len(array_2)
    return off_set

def simpleInputs(_X,_y):
    _offset =np.abs(off_set(_X,_y))
    if _offset==0:
        X = _X.as_matrix().astype(np.float)
        y=_y.astype(np.int)
    else:
        X= _X[:-_offset].as_matrix().astype(np.float)
        y = _y.astype(np.int)
    return X, y
###Main Code###
##load data
with open(_pkl_path + ticker+'_features.pickle', 'rb') as handle:
    ticker_features_ = pickle.load(handle)
with open(_pkl_path + ticker+'_labelled_data.pickle', 'rb') as handle:
    ticker_labelled_data_ = pickle.load(handle)

###the following bit is a bit redundant###
_data = ticker_labelled_data_  # data
_features = ticker_features_  # features
_keys = _data.keys()  # dates
###scalers###
mms = MinMaxScaler()
stdsc = StandardScaler()
_results=[]



from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score


list_models=fnmatch.filter(os.listdir(_models_path),str(ticker)+'*.pkl')
print list_models
T=len(list_models)
M=10
_model_results= {
    'clfs': np.empty((M, T)),
    'model_date': np.empty((M, T)),
    'data_date': np.empty((M, T)),
    'ROC': np.empty((M, T)),
    'accuracy' :np.empty((M, T))
}
for _no, _model in enumerate(list_models):
    print (_model)
    with open(os.path.join(_models_path,_model), 'rb') as handle:
        clf = pickle.load(handle)
        #_model_results['clf'][:, _no] = clf.best_score_
        _model_results['model_date'][:, _no]= _no

    for M_, _idx in enumerate(xrange(0, 9)):
        print('~~~~~~~####~~~~~~')
        print ('model date:'),_no
        print ('data date:'),_keys[_idx]
        print('~~~~~~~####~~~~~~')
        # various types of labels
        y_1 = _data[_keys[_idx]].iloc[:, 4].dropna().astype(int).values  # label_control_chart_q_30_q_70_window_5
        y_2 = _data[_keys[_idx]].iloc[:, 5].dropna().astype(int).values  # label_simple_ep_window_5_thresh_0.05- Last 4 are NaN
        y_3 = _data[_keys[_idx]].iloc[:, 6].dropna().astype(int).values  # label_simple_ep_window_5_thresh_0.00-last 4 are NaN
        duration = _data[_keys[_idx]].iloc[:, 3].dropna().astype(int).values  # can be used for regression
        short_rolling = _data[_keys[_idx]]['TradedPrice'].rolling(window=5).mean()
        long_rolling = _data[_keys[_idx]]['TradedPrice'].rolling(window=15).mean()
        ma_signal = (long_rolling - short_rolling).fillna(0)

        # feature set
        _gamma = _features[_keys[_idx]][2]
        _csi = _features[_keys[_idx]][3]
        _info = _features[_keys[_idx]][1]
        _fischer = _features[_keys[_idx]][0]
        df_new = pd.concat([_csi, _gamma, _fischer, short_rolling], axis=1).dropna()

        X, y = simpleInputs(df_new, y_3)

        print("number of classes:", len(np.unique(y)))  # of classes

        X_mms = mms.fit_transform(X)
        y_predict = clf.predict(X_mms)
#
#         print('ROC AUC: %.3f' % roc_auc_score(y, y_predict))
#         print('Accuracy: %.2f' % accuracy_score(y, y_predict))
#         print('Precision: %.3f' % precision_score(y_true=y, y_pred=y_predict))
#         print('Recall: %.3f' % recall_score(y_true=y, y_pred=y_predict))
#         print('F1: %.3f' % f1_score(y_true=y, y_pred=y_predict))
#         _model_results['clfs'][M_, _no] = int(_model.split('_')[1])
#         _model_results['data_date'][M_, _no] = _keys[_idx]
#         _model_results['ROC'][M_, _no] = roc_auc_score(y, y_predict)
#         _model_results['accuracy'][M_, _no] = accuracy_score(y, y_predict)
#
# np.save(os.path.join(_predictions_path ,str(ticker)+'_model_results.npy'), _model_results)




