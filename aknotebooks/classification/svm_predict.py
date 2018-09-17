import pandas as pd
import numpy as np
from scipy import stats
from sklearn import cluster, linear_model
import itertools as iter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as prep
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import feature_selection
from sklearn import tree
from sklearn.metrics import *
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, cross_val_score
import numpy as np

class MidpointNormalize(Normalize):
    """
    class to help renormalize the color scale
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Predictor(object):
    def __init__(self,test_data, test_labels,train_model_type="linear"):
        self.test_data_ = test_data
        self.test_labels_= test_labels

        if train_model_type == 'linear':
           # self.trained_model_ = TrainerLinear()
            pass
        elif train_model_type == 'rbf':
          #  self.trained_model_ = TrainerRBF()
            pass

class TrainerSVM:
    def __init__(self, training_data, training_labels, kernel_type):
        self.training_data_ = training_data
        self.training_labels_ = training_labels
        self.kernel_type_ = kernel_type
        self.param_grid_ = {'C': [1, 10]}  # once code is stable moving over to using this

    def hyperparam_tune(self):
        pipe_svc = Pipeline([('scl', prep.MaxAbsScaler()), ('clf', SVC(random_state=1))])
        param_range = [0.1, 1.0, 10.0, 100.0]
        param_grid = [{'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['linear', 'rbf', 'poly']}]
        gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(self.training_data_, self.training_labels_)
        tuned_param_ = gs.best_params_
        return tuned_param_

    def tuned_predictor(self):
        C_tuned_ = self.hyperparam_tune()['clf__C']
        gamma_tuned_ = self.hyperparam_tune()['clf__gamma']
        kernel_best_ = self.hyperparam_tune()['clf__kernel']
        best_clf_ = SVC(C=C_tuned_, cache_size=200, class_weight='balanced', coef0=0.0,
                            decision_function_shape=None, degree=3, gamma=gamma_tuned_, kernel=kernel_best_,
                            max_iter=-1, probability=False, random_state=None, shrinking=True,
                            tol=0.001, verbose=False)
        best_clf_.fit(self.training_data_, self.training_labels_)
        return best_clf_

    def learn_best_parameters(self):

            svc = SVC(kernel=self.kernel_type_)
            grid = GridSearchCV(svc, param_grid=self.param_grid_)
            grid.fit(self.training_data_, self.training_labels_)
            svc.fit(self.training_data_, self.training_labels_)
            print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))
            train_result_ = svc.predict(self.training_data_)
            if self.kernel_type_ == 'linear':
                print 'Linear Kernel Results'
                print classification_report(self.training_labels_, train_result_)
                print confusion_matrix(self.training_labels_, train_result_)

            elif self.kernel_type_ == 'poly':
                # svc = svm.SVC(kernel=self.kernel_type_)
                # svc.fit(self.training_data_, self.training_labels_)
                # train_result_=svc.predict(self.training_data_)
                print 'Poly Kernel Results'
                print classification_report(self.training_labels_, train_result_)
                print confusion_matrix(self.training_labels_, train_result_)
                confmat = confusion_matrix(self.training_labels_, train_result_)
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(confmat.shape[0]):
                    for j in range(confmat.shape[1]):
                        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
                plt.xlabel('predicted label')
                plt.ylabel('true label')
                plt.show()
            elif self.kernel_type_ == 'rbf':
                print 'RBF Kernel Results'
                print classification_report(self.training_labels_, train_result_)
                confmat = confusion_matrix(self.training_labels_, train_result_)
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(confmat.shape[0]):
                    for j in range(confmat.shape[1]):
                        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
                plt.xlabel('predicted label')
                plt.ylabel('true label')
                plt.show()

    def sample_predict(self, clf):

        test_result = clf.predict(self.test_data_)
        print classification_report(self.test_labels_, test_result)
        print confusion_matrix(self.test_labels_, test_result)

    def new_predict(self, clf): #this needs cleaning up dont use just yet
        test_result = clf.predict(test_data)
        with open('result/predict_label.txt', 'wt') as f:
            for i in range(len(test_result)):
                f.writelines(test_result[i])
        self.test_target = test_result
        print 'write over'



#         def cross_validation(self):
#                     scores = cross_val_score(self.clf, self.training_data, self.training_labels, cv=cv, scoring='f1_macro')
#                     print scores
#                     print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#                     #print("The best parameters are %s with a score of %0.2f"
#                      #       % (grid.best_params_, grid.best_score_))
#                     self.draw_visualization_param_effect(grid, C_range, gamma_range)
#
#     def draw_visualization_param_effect(self, grid, C_range, gamma_range):
#         scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
#                                                              len(gamma_range))
#         plt.figure(figsize=(8, 6))
#         plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
#         plt.imshow(scores, interpolation='nearest',
#                    norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
#         plt.xlabel('gamma')
#         plt.ylabel('C')
#         plt.colorbar()
#         plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
#         plt.yticks(np.arange(len(C_range)), C_range)
#         plt.title('Validation accuracy')
#         plt.savefig('param_effect.png')
#         plt.show()
#
#     def train_classifier(self, training_data, training_labels):
#         self.clf.fit(training_data, training_labels)
#         training_result = self.clf.predict(self.training_data)
#         print classification_report(training_labels, training_result)
#
#
#     def cross_validation(self):
#         cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
#         scores = cross_val_score(self.clf, self.training_data, self.training_labels, cv=cv, scoring='f1_macro')
#         print scores
#         print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
#
#
#
#
#
# def SVM_train(train_data, train_labels):
#     clf = SVC(kernel='linear', class_weight='balanced', C =100, gamma = 0.01)
#     clf.fit(train_data, train_labels)
#     expected = train_labels
#     predicted = clf.predict(train_data)
#     # summarize the fit of the model
#     print classification_report(expected, predicted)
#     print confusion_matrix(expected, predicted)
#
#
# def feature_selection(data, data_labels, feature_names):
#     clf = SVC(class_weight='auto', C =2)
#     clf.fit(data, data_labels)
#
#
#
#
#
#
