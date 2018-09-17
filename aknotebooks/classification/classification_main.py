import pickle
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Added version check for recent scikit-learn 0.18 checksok
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter
from time import time
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.learning_curve import learning_curve
import seaborn as sns
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

# to make this notebook's output stable across runs
np.random.seed(42)
stdsc = StandardScaler()
mms = MinMaxScaler()
plt.interactive(False)
#_test_path = #placeholder for path

_test_path =('/home/ak/Documents/features_test/')

###functions that are needed###
def scaler_comparison(_X, _y, C_, _scaler_type, _test_size):
    lr = LogisticRegression(penalty='l1', C=C_)
    ##get training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(_X, _y, test_size=_test_size, random_state=0)
    ##normalise
    X_train_norm = _scaler_type.fit_transform(X_train)
    X_test_norm = _scaler_type.transform(X_test)
    ##fit model
    lr_norm = lr.fit(X_train_norm, y_train)

    lr_score_train = lr_norm.score(X_train_norm, y_train)
    lr_score_test = lr_norm.score(X_test_norm, y_test)

    return lr_score_train, lr_score_test  # returns Train Accuracy, returns Test Accuracy

def split_data(_X,_y, _test_size):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=_test_size, random_state=0)
    return X_train, X_test, y_train, y_test

def weights_coefficients(_X,_y):
    weights, params = [], []
    for c in np.arange(-4., 6.):
        lr = LogisticRegression(penalty='l1', C=10. ** c, random_state=0)
        lr.fit(_X,_y)
        weights.append(lr.coef_[1])
        params.append(10. ** c)
    weights = np.array(weights)
    return weights


def plot_weights_coefficients(df_, weights):
    fig = plt.figure()
    ax = plt.subplot(111)

    colors = ['blue', 'green', 'red', 'cyan',
              'magenta', 'yellow', 'black',
              'pink', 'lightgreen', 'lightblue',
              'gray', 'indigo', 'orange']
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column],
                 label=df_.columns[column],
                 color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.title('Weight Coefficient versus Regularisation Parameter')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.38, 1.03),
              ncol=1, fancybox=True)
    plt.show()


# Sequential Backward Selection (SBS)
# adapted from ML Book/Raschka
class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def accuracy_features(_X, _y):
    #need to adjust max ylim and min ylim
    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=1) # selecting features using SBS
    sbs.fit(_X, _y)
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    # plt.savefig('./sbs.png', dpi=300)
    plt.show()

def accuracy_components(_X, _y, _clf_):
    train_sizes, train_scores, test_scores =\
                learning_curve(estimator=_clf_,
                               X=_X,
                               y=_y,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    return train_sizes, train_mean, train_std, test_mean, test_std

def accuracy_plot(train_sizes, train_mean, train_std, test_mean, test_std):
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')


    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    # plt.savefig('./figures/learning_curve.png', dpi=300)
    plt.show()









# plotting performance of feature subsets
