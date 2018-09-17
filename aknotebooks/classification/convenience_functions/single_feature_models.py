import pickle
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

##useful functions##




###ticker###
ticker='HSBA.L' #enter symbol you want - this will be made for various

# paths
_pkl_path =('/home/ak/Documents/features_models/')#pkl path
_graphs_path= os.path.join(_pkl_path, 'graphs')
_models_path = ('/home/ak/Documents/features_test/models/')

###predicitions path
_predictions_path ='/home/ak/Documents/features_models/predictions/'

###main code###

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
_clfs=[]
_xr=len(_keys)-1
print("#of models:",_xr)
print("for ticker:", ticker)
for _idx in xrange(0, _xr):
    # various types of labels
    y_1 = _data[_keys[_idx]].iloc[:, 4].dropna().astype(int).values  # label_control_chart_q_30_q_70_window_5
    y_2 = _data[_keys[_idx]].iloc[:, 5].dropna().astype(
        int).values  # label_simple_ep_window_5_thresh_0.05- Last 4 are NaN
    y_3 = _data[_keys[_idx]].iloc[:, 6].dropna().astype(
        int).values  # label_simple_ep_window_5_thresh_0.00-last 4 are NaN
    duration = _data[_keys[_idx]].iloc[:, 3].dropna().astype(int).values  # can be used for regression

    # feature set
    _gamma = _features[_keys[_idx]][2]
    _csi = _features[_keys[_idx]][3]
    _info = _features[_keys[_idx]][1]
    _fischer = _features[_keys[_idx]][0]
    short_rolling = _data[_keys[_idx]]['TradedPrice'].rolling(window=5).mean()
    long_rolling = _data[_keys[_idx]]['TradedPrice'].rolling(window=15).mean()
    ma_signal = (long_rolling - short_rolling).fillna(0)
    df_new = pd.concat([_csi, _gamma, _fischer, short_rolling], axis=1).dropna()

    X, y = simpleInputs(df_new, y_3)

    print("number of classes:",len(np.unique(y)))  # of classes

    X_mms = X = mms.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_mms, y, test_size=.1, random_state=0)

    #     # # Train a SVM classification model
    param_grid = dict(kernel=["rbf"],
                      C=[1,5,10,25,50],
                      gamma=[0.0001,0.001,0.01])

    clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, verbose=1,n_jobs=-1,cv=5)

    clf.fit(X_train, y_train)
    filename = _models_path + str(ticker) + '_' + str(_keys[_idx]) + '_clf' + '.joblib.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print('SVM Best Params & Score:')
    print clf.best_params_, clf.best_score_
    _clfs.append(clf)
    clf_files = _models_path + str(ticker) + '_clfs_'  + '.pkl'
    pickle.dump(_clfs, open(clf_files, 'wb'))
    print("#of models   left:",_xr -_idx
