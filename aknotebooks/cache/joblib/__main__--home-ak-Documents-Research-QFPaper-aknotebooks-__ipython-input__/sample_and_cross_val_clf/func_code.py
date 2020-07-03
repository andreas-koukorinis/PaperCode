# first line: 1
def sample_and_cross_val_clf(train_size=200, noise_corr=2, dim=3, sep=.5,
                             random_state=0):
    clf = LinearSVC(penalty='l2', fit_intercept=True)

    n_samples = train_size + 10000
    X, y = mk_data(n_samples=n_samples,
                   separability=sep, random_state=random_state,
                   noise_corr=noise_corr, dim=dim)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    validation_score = accuracy_score(
                            y_test,
                            clf.fit(X_train, y_train).predict(X_test))

    scores = dict()
    scores['loo'] = (np.mean(cross_val_score(clf, X_train, y_train,
                            cv=LeaveOneOut(train_size)))
                        - validation_score)

    # Create 10 blocks of evenly-spaced labels for LabelShuffleSplit
    labels = np.arange(train_size) // (train_size // 10)

    scores['3 splits'] = (np.mean(cross_val_score(clf, X_train, y_train,
                    cv=LabelShuffleSplit(labels, n_iter=3, random_state=0)))
                - validation_score)
    scores['10 splits'] = (np.mean(cross_val_score(clf, X_train, y_train,
                    cv=LabelShuffleSplit(labels, n_iter=10, random_state=0)))
                - validation_score)
    scores['200 splits'] = (np.mean(cross_val_score(clf, X_train, y_train,
                    cv=LabelShuffleSplit(labels, n_iter=200, random_state=0)))
                - validation_score)

    return scores
