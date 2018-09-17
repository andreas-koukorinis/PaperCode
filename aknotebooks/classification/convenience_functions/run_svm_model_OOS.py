from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas as pd
from sklearn.externals import joblib
# Added version check for recent scikit-learn 0.18 checksok ca
from distutils.version import LooseVersion as Version
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import pickle
from sklearn.svm import SVC # "Support Vector Classifier"
import matplotlib.pyplot as plt
from math import sqrt
###paths###

# Where to save the figures
_pkl_path ='/home/ak/Documents/Data/features_models/'

_graphs_= os.path.join(_pkl_path, 'graphs')
_models_ = os.path.join(_pkl_path, 'models')
_predictions_ =os.path.join(_pkl_path, 'predictions')
_figures_= ('/home/ak/Documents/Research/QFPaper/paper/figures/')


###predicitions path
_predictions_path ='/home/ak/Documents/features_models/predictions/'

sys.path.append('/home/ak/Documents/Research/hsmm/')

mms=MinMaxScaler()
###useful functions to be replaced by import###
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
def ensure_dir(file_path): #ensure a dictory exists otherwise create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print('New Directory Created',file_path)
    else:
        print('Directory Exists!check')
def fwd_dates(_dates_list, _key_date):
    #returns a list of dates that are forward from the key_date
    fwd_dates_list  = [i for i in _dates_list if i > _key_date]
    return fwd_dates_list

def prec_recall_report(y_true_, y_predict_):
    #function to ge the sci-kit learn classification metrics into a pretty DF for csv!
    report = pd.DataFrame(list(precision_recall_fscore_support(y_true_, y_predict_)),
                          index=['Precision', 'Recall', 'F1-score', 'Support']).T
    # Now add the 'Avg/Total' row
    report.loc['Avg/Total', :] = precision_recall_fscore_support(y_cc, y_predict, average='weighted')
    report.loc['Avg/Total', 'Support'] = report['Support'].sum()
    return report
#####
symbols_ftse= ['ECM','REL','RTO','UU.','NG','PRU','DMG']

###ticker###

for ticker in symbols_ftse: #ticker = 'VOD' ##load symbols/data
    model_ticker_path =os.path.join(_models_+'/'+ticker) #path where all your models are saved
#### load files#####
    with open(_pkl_path + ticker+'_features.pickle', 'rb') as handle:
        ticker_features_ = pickle.load(handle)
    with open(_pkl_path + ticker+'_labelled_data.pickle', 'rb') as handle:
        ticker_labelled_data_ = pickle.load(handle)
    ###list of dates for your symbol###
    _dates = ticker_labelled_data_.keys()

    ###load first model- change the 0 to iterate across _dates list


    ###load model
    for key_, date_ in enumerate(_dates):
        print key_, date_

        model_to_load=str(_dates[key_])+'_svm_.pkl'
        svm_clf_ = os.path.join(model_ticker_path,model_to_load)
        ticker_predictions_path_ = os.path.join(_predictions_, ticker)
        ensure_dir(ticker_predictions_path_)
        if os.path.isfile(svm_clf_) == True:
            with open(svm_clf_, 'rb') as handle:
                estimator = pickle.load(handle)

            fwd_dates = [i for i in _dates if i > _dates[key_]]
            for _fwd_idx, _fwd_date in enumerate(fwd_dates):

                y_cc = ticker_labelled_data_[fwd_dates[_fwd_idx]].iloc[:, 6].dropna().astype(int).values
                # feature set
                _gamma = ticker_features_[fwd_dates[_fwd_idx]][2]
                _csi = ticker_features_[fwd_dates[_fwd_idx]][3]
                _info = ticker_features_[fwd_dates[_fwd_idx]][1]
                _fischer = ticker_features_[fwd_dates[_fwd_idx]][0]
                df_features = pd.concat([_csi, _gamma, _fischer], axis=1)

                df_features['MA_signal'] = \
                    ticker_labelled_data_[fwd_dates[_fwd_idx]]['TradedPrice'].rolling(window=15).mean() \
                    - ticker_labelled_data_[fwd_dates[_fwd_idx]]['TradedPrice'].rolling(window=5).mean()
                df_features = df_features.dropna()

                X, y = simpleInputs(df_features, y_cc)
                X_mms = mms.fit_transform(X)
                print(len(X_mms) == len(y))
                y_predict = estimator.predict(X_mms)
                clf_report = prec_recall_report(y_cc, y_predict)
                print clf_report
                print('Accuracy: %.2f' % accuracy_score(y_cc, y_predict))
                report_name = ticker_predictions_path_ +'/'+ticker+'_'+str(_fwd_date)+'_class_report.csv'
                clf_report.to_csv(report_name)
        else:
            print "no model available"
