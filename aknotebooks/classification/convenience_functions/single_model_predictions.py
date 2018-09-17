import pickle
import pandas as pd
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import svm

mms = MinMaxScaler()
stdsc = StandardScaler()

sys.path.append('/home/ak/Documents/Research/QFPaper/Huber-Norm-Regularizer')

from huber_svm import HuberSVC


def off_set(self, array_1, array_2):
    off_set = len(self.array_1) - len(self.array_2)
    return off_set


def simpleInputs(self, _X, _y):
    _offset = np.abs(self.off_set(_X, _y))
    if _offset == 0:
        X = self._X.as_matrix().astype(np.float)
        y = self._y.astype(np.int)
    else:
        X = self._X[:-_offset].as_matrix().astype(np.float)
        y = self._y.astype(np.int)
    return X, y

class DataFuncs():
    def __init__(self,_pkl_path,_graphs_path,_models_path,_predictions_path, ticker='C.N'):
        # Where to save the figures
        self._pkl_path = ('/home/ak/Documents/features_test/')
        self.ticker = ticker  # enter symbol you want - this will be made for variou
        self._graphs_path = '/home/ak/Documents/features_test/graphs/'
        self._models_path = '/home/ak/Documents/features_test/models/'
        self._predictions_path = '/home/ak/Documents/features_test/predictions/'

##load data
    def load_data(self):
        with open(self._pkl_path + self.ticker+'_features.pickle', 'rb') as handle:
            ticker_features_ = pickle.load(handle)
        with open(self._pkl_path + self.ticker+'_labelled_data.pickle', 'rb') as handle:
            ticker_labelled_data_ = pickle.load(handle)

        return ticker_features_, ticker_labelled_data_
    def save_data(self):
        pass

lasso = OneVsRestClassifier(Lasso())
param_lasso = {'estimator__alpha': [100, 10, 1, 0.1, 1e-2, 1e-3]}

elastic = OneVsRestClassifier(ElasticNet())
param_elastic = {'estimator__alpha': [100, 10, 1, 0.1, 1e-2, 1e-3],
                 'estimator__l1_ratio': np.linspace(0.01, 0.99, 5)}

ridge = RidgeClassifier(solver='lsqr')
param_ridge = {'alpha': [100, 10, 1, 0.1, 1e-2, 1e-3]}

huber = OneVsRestClassifier(HuberSVC())
param_huber = {'estimator__C': [100, 10, 1, 0.1, 1e-2, 1e-3],
              'estimator__lambd': [100, 10, 1, 0.1, 1e-2, 1e-3],
              'estimator__mu': [100, 10, 1, 0.1, 1e-2, 1e-3]}

param_lr = {
        'C':  np.arange(1.0, 3.0)
         , 'solver': ['newton-cg']
    }
lr_clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)

param_svc = dict(kernel=["linear","rbf"],C=[1,5,10,25,50],gamma=[0.0001,0.001,0.01])
svc_clf = svm.SVC(class_weight='balanced')
###see if the below works###
n_folds = 5
param_folds = 3
scoring = 'accuracy'

result_df = pd.DataFrame()
best_clfs = []

cv=StratifiedKFold(n_splits=n_folds)

for i, (train_index, test_index) in enumerate(cv.split(X,y)):
    for clf_name, clf, param_grid in [('Lasso', lasso, param_lasso),
                                      ('ElasticNet', elastic, param_elastic),
                                      ('Ridge', ridge, param_ridge),
                                      ('Logistic', lr_clf, param_lr),
                                      ('SVC', svc_clf, param_svc),
                                      ('HuberSVC', huber, param_huber)]:


        gs = GridSearchCV(clf, param_grid, scoring=scoring, cv=param_folds, n_jobs=-1, refit='AUC')
        gs.fit(X[train_index], y[train_index])
        best_clf = gs.best_estimator_
        best_clfs.append(best_clf)

        score = accuracy_score(y[test_index], best_clf.predict(X[test_index]))
        result_df.loc[i, clf_name] = score