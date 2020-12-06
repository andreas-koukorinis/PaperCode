import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, multilabel_confusion_matrix, f1_score, \
    average_precision_score, roc_auc_score, recall_score
from sklearn.multiclass import OneVsRestClassifier  # support from multiclass
from sklearn.svm import SVC


class FitModels(object):
    # this needs documentation!

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

    def best_svc_clf(self):
        param_grid = dict(kernel=['linear', 'rbf'],
                          C=[1, 10, 100, 1000],
                          gamma=[0.0001, 0.001, 0.01, 0.02])

        clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, verbose=1, n_jobs=-1, cv=11)
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


class MetricsResults(object):
    """
    input correct and predicted labels and get model results
    y_true: array-like of shape (n_samples,) or (n_samples, n_classes)
    True labels or binary label indicators.
    y_score: array-like of shape (n_samples,) or (n_samples, n_classes) target scores.
    In the binary and multilabel cases, these can be either probability estimates or non-thresholded decision values (as returned by decision_function on some classifiers).
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def model_results(self):
        # The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It
        # is defined as the average of recall obtained on each class.

        model_results_dict = {
            'recall score': recall_score(self.y_true, self.y_pred, average='macro'),
            'Weighted average precision score': average_precision_score(self.y_true, self.y_pred),
            'Balanced accuracy score': balanced_accuracy_score(self.y_true, self.y_pred),
            'f1_score': f1_score(self.y_true, self.y_pred, average='weighted'),
            'micro f1 score': f1_score(self.y_true, self.y_pred, average='micro'),
            'MultiLabel confusion matrix': multilabel_confusion_matrix(self.y_true, self.y_pred),

        }
        return model_results_dict

    def roc_auc_score(self, y_scores):
        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        output = roc_auc_score(self.y_true, y_scores)
        return output
