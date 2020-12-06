import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, multilabel_confusion_matrix, f1_score, \
    average_precision_score, roc_auc_score, recall_score
from sklearn.multiclass import OneVsRestClassifier  # support from multiclass
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit



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


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

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

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def output_plot_learning_curve(estimator, X, y, title= None):
    # this one may need a bit of refactoring
    fig, axes = plt.subplots(3, 2, figsize = (10, 15))
    title = r"Learning Curves (SVC, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)

    plt.show()

