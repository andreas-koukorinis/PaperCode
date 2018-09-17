__author__ = 'ak'
import sys
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from scipy import stats
from sklearn import cluster
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, decomposition
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import scipy as scp

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn import grid_search

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from sklearn.datasets import make_circles
from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import itertools
import time
plt.style.use('fivethirtyeight') # Good looking plots
pd.set_option('display.max_columns', None) # Display any number of columns
import matplotlib.pyplot as plt
import pylab as pl
from collections import OrderedDict
import math
import datetime
from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn
import numpy as np
import pandas as pd
from collections import OrderedDict
import math
from sklearn.linear_model import LogisticRegression

import datetime
import sys

from matplotlib.colors import ListedColormap

from sklearn.cross_validation import train_test_split
sys.path.append('/home/ak/Documents/Research/master/hsmm/')

# To plot pretty figures

import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

##useful functions###


def ensure_dir(file_path): #ensure a dictory exists otherwise create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('New Directory Created',file_path)
    else:
        print('Directory Exists!check')

def symbol_path_(_main_path, _symb): #create directory for each symbol!
    _main_path = _ftse
    _symbol_path = os.path.join(_main_path,_symb+'/')
    print(_symbol_path)
    ensure_dir(_symbol_path)
    return _symbol_path

def symbol_features_path(_main_path, _symb):
    _symb_features_path = os.path.join(_main_path,_symb+'/')
    print(_symb_features_path)
    ensure_dir(symbol_features_path)
    return symbol_features_path


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




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
def no_nan(_array):
    no_nan= _array.isnull().sum()
    return no_nan

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred

def rmse_scorer(clf, y_actual, y_predicted):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rms = sqrt(mean_squared_error(y_actual[1:], y_predicted[:-1]))
    return -(rms)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_weights_coefficients(df_):
    #to--do: fix the variables and this function
    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = plt.subplot(111)

    # colors = ['blue', 'green', 'red', 'cyan',
    #           'magenta', 'yellow', 'black',
    #           'pink', 'lightgreen', 'lightblue',
    #           'gray', 'indigo', 'orange']

    # weights, params = [], []
    # for c in np.arange(-4., 6.):
    #     lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    #     lr.fit(X_train, y_train)
    #     weights.append(lr.coef_[1])
    #     params.append(10.**c)

    # weights = np.array(weights)
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
    ###plotting functions for various pieces##
    def show_values(pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857
        By HYRY
        '''
        from itertools import izip
        pc.update_scalarmappable()
        ax = pc.get_axes()
        for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    def cm2inch(*tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: https://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)

    def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
                correct_orientation=False, cmap='RdBu'):
        '''
        Inspired by:
        - https://stackoverflow.com/a/16124677/395857
        - https://stackoverflow.com/a/25074150/395857
        '''

        # Plot it out
        fig, ax = plt.subplots()
        # c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
        c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

        # set tick labels
        # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Remove last blank column
        plt.xlim((0, AUC.shape[1]))

        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell
        show_values(c)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            # resize
        fig = plt.gcf()
        # fig.set_size_inches(cm2inch(40, 20))
        # fig.set_size_inches(cm2inch(40*4, 20*4))
        fig.set_size_inches(cm2inch(figure_width, figure_height))

    def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
        '''
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857
        '''
        lines = classification_report.split('\n')

        classes = []
        plotMat = []
        support = []
        class_names = []
        for line in lines[2: (len(lines) - 2)]:
            t = line.strip().split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            print(v)
            plotMat.append(v)

        print('plotMat: {0}'.format(plotMat))
        print('support: {0}'.format(support))

        xlabel = 'Metrics'
        ylabel = 'Classes'
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
        figure_width = 25
        figure_height = len(class_names) + 7
        correct_orientation = False
        heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
                correct_orientation, cmap=cmap)


