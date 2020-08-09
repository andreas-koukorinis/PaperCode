import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm import tqdm
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

try:
    import pandas as pd
except:
    pass

def fit_distributions(X, distributions=None, verbose=False):
    """
    fit distributions to data `X`
    Parameters
    ----------
    X : 1d-array
    distributions : array of strings (['norm', 'laplace', etc..])
    defaults distributions are: ['norm', 'laplace', 'cauchy']
    verbose: bool, default False
    """

    if distributions is None:
        if min(X) >= 0:
            distributions = [
                'halfnorm', 'halfcauchy'
            ]
        else:
            distributions = [
                'norm', 'laplace', 'cauchy'
            ]

    params = {}
    for name in distributions:
        distr = getattr(stats, name)
        params[name] = distr.fit(X)
        if verbose:
            print(name, params[name])
    return params


def plot(X, symbolTitle=None, X_name=None, params=None, **kwargs):
    """
    plot probability distribution function for `y`
    and overlay distributions calculated with `params`
    Parameters
    ----------
    x : array
    params: list of best-fit parameters returned by fit_distributions() function
    Return value
    ------------
    params from fit_distributions() function
    """

    if X is not np.ndarray:
        X = np.array(X)
    if params is None:
        print("Estimating distributions parameters...")
        params = fit_distributions(X, verbose=True)

    label = X_name or "data"

    # plot PDF
    x_min = np.percentile(X, 0.9)
    x_max = -np.percentile(-X, 0.9)
    X_ = X[(X>=x_min) & (X<=x_max)]
    num_bins = int(np.log(len(X_))*5)
    x_space = np.linspace(x_min, x_max, 1000)

    f, ax = plt.subplots(**kwargs)
    ax.hist(X_, bins=num_bins, density=True, alpha=0.33, color="dodgerblue", label=label)
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(x_space, distr.pdf(x_space, loc=param[0], scale=param[1]), label=name)
        ax.set_xlabel('values', fontsize=5)
        ax.set_ylabel('pdf', fontsize='small')
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'MicroPriceReturnsLongTail' + str(symbolTitle)
                    +'plot2'+ '.png', dpi=150)

    ax.legend()
    ax.set_ylabel('pdf', fontsize='small')
    if X_name is not None:
        ax.set_xlabel(X_name, fontsize=5)
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'MicroPriceReturnsLongTail' + str(symbolTitle)
                    + str(X_name) + '.png', dpi=150)
    ax.grid(True)
    plt.show()

    # plot LOG PDF
    x_min, x_max = X.min(), X.max()

    num_bins = int(np.log(len(X))*5)
    x_space = np.linspace(x_min, x_max, 1000)

    bins_means = []  # mean of bin interval
    bins_xs = []  # number of ys in interval

    x_step = (x_max - x_min) / num_bins
    for x_left in np.arange(x_min, x_max, x_step):
        bins_means.append(x_left + x_step/2.)
        bins_xs.append(np.sum((X>=x_left) & (X<x_left+x_step)))
    bins_xs = np.array(bins_xs) / len(X) / x_step  # normalize

    f, ax = plt.subplots(**kwargs)
    ax.scatter(bins_means, bins_xs, s=5., color="dodgerblue", label=label)
    for name, param in params.items():
        distr = getattr(stats, name)
        ax.plot(x_space, distr.pdf(x_space, loc=param[0], scale=param[1]), label=name)
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'MicroPriceReturnsPDF' + str(symbolTitle)
                    + '.png', dpi=150)

    ax.legend()
    ax.set_ylabel('pdf', fontsize=5)
    ax.set_yscale('log')
    if X_name is not None:
        ax.set_xlabel(X_name, fontsize=5)
        plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'MicroPriceReturnsLogPDF' + str(symbolTitle)
                    +str(X_name)+ '.png', dpi=150)
    ax.grid(True)
    plt.show()

    return params


class GaussianScaler():
    """Transform data to make it Gaussian distributed."""

    def __init__(self):
        self.transform_table = None
        self.features_names = None
        self.__num_vars = None

    def fit(self, X, y=None):
        """Compute empirical parameters for transforming the data to Gaussian distribution.
        Parameters:
        -----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to fit
        """

        if len(X.shape)>2:
            raise NotImplementedError("X must be an 1d-array or a 2d-matrix of observations x features")

        # convert from pd.DataFrame to np.ndarrray:
        if "pandas.core.frame" in sys.modules.keys() and \
            type(X) in (pd.core.series.Series, pd.core.frame.DataFrame):
            if type(X) == pd.core.frame.DataFrame:
                self.features_names = X.columns.values
            else:
                self.features_names = np.array([X.name])
            X = X.values

        if X.dtype not in (float, np.float32, np.float64):
            raise Exception("X.dtype is {}, but should be float".format(X.dtype))

        if len(X.shape) == 2:
            if X.shape[1] == 1:
                X = X.ravel()
                self.__num_vars = 1
            else:
                self.__num_vars = X.shape[1]
        else:
            self.__num_vars = 1

        self.transform_table = []

        for j in range(self.__num_vars):
            if self.__num_vars > 1:
                X_sorted = np.array(np.unique(X[:, j], return_counts=True)).T
            else:
                X_sorted = np.array(np.unique(X, return_counts=True)).T
            X_sorted[:, 1] = np.cumsum(X_sorted[:, 1])
            total = X_sorted[-1,1]
            # X_sorted[:, 0] is x, X_sorted[:, 1] is the number of occurences <= x

            STEP_MULT = 0.1  # step multiplier
            MIN_STEP = 5

            step = MIN_STEP
            i = step
            prev_x = -np.inf

            transform_table = []
            transform_table.append((-np.inf, -np.inf, 0.))

            while True:
                index = np.argmax(X_sorted[:,1] >= i)
                row = X_sorted[index]
                x = row[0]
                if x != prev_x:
                    cdf_empiric = row[1] / total
                    x_norm = stats.norm.ppf(cdf_empiric)

                    if x_norm == np.inf:  # too large - stop
                        break
                    if x_norm != -np.inf:
                        transform_table.append((x, x_norm, 0.))

                    if cdf_empiric < 0.5:
                        step = int(row[1] * STEP_MULT)
                    else:
                        step = int((total - row[1]) * STEP_MULT)

                    step = max(step, MIN_STEP)
                    prev_x = x

                i = i + step
                if i >= total:
                    break

            transform_table.append((np.inf, np.inf, 0.))
            transform_table = np.array(transform_table)

            # compute x -> x_norm coefficients
            dx = transform_table[2:-1, 0] - transform_table[1:-2, 0]
            dx_norm = transform_table[2:-1, 1] - transform_table[1:-2, 1]
            transform_table[2:-1, 2] = dx_norm / dx

            """
            # generic non-optimized code would look like this:
            for i in range(2, len(transform_table) - 1):
                dx = transform_table[i, 0] - transform_table[i-1, 0]
                dx_norm = transform_table[i, 1] - transform_table[i-1, 1]
                transform_table[i, 2] = dx_norm / dx
            """

            # fill boundary bins (plus/minus infinity) intervals:
            transform_table[0,  2] = transform_table[2,  2]
            transform_table[1,  2] = transform_table[2,  2]
            transform_table[-1, 2] = transform_table[-2, 2]

            # add current transform table for the feature to self.transform_table
            self.transform_table.append(transform_table)

    def transform(self, X, y=None):
        """Transform X to Gaussian distributed (standard normal).
        Parameters
        ----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to transform
        """

        if self.transform_table is None:
            raise Exception(("This GaussianScaler instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this method."))

        if len(X.shape)>2:
            raise NotImplementedError("X must be an 1d-array or a 2d-matrix of observations x features")

        # convert from pd.DataFrame to np.ndarrray:
        if "pandas.core.frame" in sys.modules.keys() and \
            type(X) in (pd.core.series.Series, pd.core.frame.DataFrame):
            if type(X) == pd.core.frame.DataFrame:
                features_names = X.columns.values
            else:
                features_names = np.array([X.name])

            if (features_names != self.features_names).any():
                raise Exception("Feature names mismatch.\nFeatures for fit():{}\nFeatures for transform:{}".format(
                    self.features_names, features_names))
            save_index = X.index.copy()
            X = X.values.copy()

        if X.dtype not in (float, np.float32, np.float64):
            raise Exception("X.dtype is {}, but should be float".format(X.dtype))

        if len(X.shape) == 2:
            if X.shape[1] == 1:
                X = X.ravel()
                num_vars = 1
            else:
                num_vars = X.shape[1]
        else:
            num_vars = 1

        if self.__num_vars != num_vars:
            raise Exception("Number of features mismatch for fit() and transform(): {} vs {}".format(
                self.__num_vars, num_vars))

        def _transform(x, j):
            # x(empirical) -> x(normaly distributed)
            transform_table = self.transform_table[j]
            lefts  = transform_table[transform_table[:, 0] <  x]
            rights = transform_table[transform_table[:, 0] >= x]

            left_boundary = lefts[-1]
            right_boundary = rights[0]

            k = right_boundary[2]

            if right_boundary[0] == np.inf:
                x_norm = left_boundary[1] + k * (x - left_boundary[0])
            else:
                x_norm = right_boundary[1] + k * (x - right_boundary[0])

            return x_norm

        vtransform = np.vectorize(_transform)

        if num_vars > 1:
            # transform all features:
            for j in range(self.__num_vars):
                X[:, j] = vtransform(X[:, j], j)
        else:
            X = vtransform(X, 0)

        # reconstruct X as a Series or a DataFrame:
        if self.features_names is not None:
            if num_vars > 1:
                X = pd.DataFrame(X, columns=self.features_names, index=save_index)
            else:
                X = pd.Series(X, name=self.features_names[0], index=save_index)

        return X

    def fit_transform(self, X, y=None):
        """ Fit to data, then transform it.
        Parameters
        ----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to transform
        """

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Transform back the data from Gaussian to the original empirical distribution.
        Parameters
        ----------
        X : array-like (1 or 2 dim. np.ndarray, pandas.Series or pandas.DataFrame)
        features to inverse transform
        """

        if self.transform_table is None:
            raise Exception(("This GaussianScaler instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this method."))

        if len(X.shape)>2:
            raise NotImplementedError("X must be an 1d-array or a 2d-matrix of observations x features")

        # convert from pd.DataFrame to np.ndarrray:
        if "pandas.core.frame" in sys.modules.keys() and \
            type(X) in (pd.core.series.Series, pd.core.frame.DataFrame):
            if type(X) == pd.core.frame.DataFrame:
                features_names = X.columns.values
            else:
                features_names = np.array([X.name])

            if (features_names != self.features_names).any():
                raise Exception("Feature names mismatch.\nFeatures for fit():{}\nFeatures for transform:{}".format(
                    self.features_names, features_names))
            save_index = X.index.copy()
            X = X.values.copy()

        if X.dtype not in (float, np.float32, np.float64):
            raise Exception("X.dtype is {}, but should be float".format(X.dtype))

        if len(X.shape) == 2:
            if X.shape[1] == 1:
                X = X.ravel()
                num_vars = 1
            else:
                num_vars = X.shape[1]
        else:
            num_vars = 1

        if self.__num_vars != num_vars:
            raise Exception("Number of features mismatch for fit() and transform(): {} vs {}".format(
                self.__num_vars, num_vars))

        def _inverse_transform(x, j):
            # x(normaly distributed) -> x(empirical)
            transform_table = self.transform_table[j]
            lefts  = transform_table[transform_table[:, 1] <  x]
            rights = transform_table[transform_table[:, 1] >= x]

            left_boundary = lefts[-1]
            right_boundary = rights[0]

            k = right_boundary[2]

            if right_boundary[1] == np.inf:
                x_emp = left_boundary[0] + (x - left_boundary[1]) / k
            else:
                x_emp = right_boundary[0] + (x - right_boundary[1]) / k

            return x_emp

        vinverse_transform = np.vectorize(_inverse_transform)

        if num_vars > 1:
            # transform all features:
            for j in range(self.__num_vars):
                X[:, j] = vinverse_transform(X[:, j], j)
        else:
            X = vinverse_transform(X, 0)

        # reconstruct X as a Series or a DataFrame:
        if self.features_names is not None:
            if num_vars > 1:
                X = pd.DataFrame(X, columns=self.features_names, index=save_index)
            else:
                X = pd.Series(X, name=self.features_names[0], index=save_index)

        return X


def sample_hsic(n, angle, sigma=0.2, offset=1):
    n4 = int(n / 4)
    N = np.random.randn(n4, 2) * sigma
    S = np.random.randn(n4, 2) * sigma
    E = np.random.randn(n4, 2) * sigma
    W = np.random.randn(n4, 2) * sigma

    N[:, 1] += offset
    S[:, 1] -= offset
    W[:, 0] -= offset
    E[:, 0] += offset

    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    A = R.dot(np.vstack((N, S, W, E)).T).T

    return A[:, 0], A[:, 1]


def simple_statistic(X, Y, squared=False):
    assert X.ndim == Y.ndim == 1

    # IMPLEMENT: compute mean difference of X and Y
    result = np.mean(X) - np.mean(Y)

    if squared:
        result *= result
    return result


def two_sample_permutation_test(test_statistic, X, Y, num_permutations, prog_bar=True):
    assert X.ndim == Y.ndim

    statistics = np.zeros(num_permutations)

    range_ = range(num_permutations)
    if prog_bar:
        range_ = tqdm(range_)
    for i in range_:
        # concatenate samples
        if X.ndim == 1:
            Z = np.hstack((X, Y))
        elif X.ndim == 2:
            Z = np.vstack((X, Y))

        # IMPLEMENT: permute samples and compute test statistic
        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        my_test_statistic = test_statistic(X_, Y_)
        statistics[i] = my_test_statistic
    return statistics

def plot_permutation_samples_sns(null_samples, symbolTitle, statistic= None):
    sns.distplot(null_samples)
    plt.axvline(x=np.percentile(null_samples, 2.5), ls='--', c='b')
    legend = ["95% quantiles"]
    if statistic is not None:
        plt.axvline(x=statistic,ls='--',  c='r')
        legend += ["Actual test statistic"]
    plt.legend(legend)
    plt.axvline(x=np.percentile(null_samples, 97.5),ls='--',  c='b')
    plt.xlabel("Test statistic value")
    plt.ylabel("Counts")
    plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'seaborn_plot_permutation_samples' + str(symbolTitle)
                + '.png', dpi=150)

def plot_permutation_samples(null_samples,symbolTitle, statistic=None):
    plt.hist(null_samples)
    plt.axvline(x=np.percentile(null_samples, 2.5), c='b')
    legend = ["95% quantiles"]
    if statistic is not None:
        plt.axvline(x=statistic,ls='--',  c='r')
        legend += ["Actual test statistic"]
    plt.legend(legend)
    plt.axvline(x=np.percentile(null_samples, 97.5), c='b')
    plt.xlabel("Test statistic value")
    plt.ylabel("Counts")
    plt.savefig('/home/ak/Documents/Research/Papers/figures/' + 'plot_permutation_samples' + str(symbolTitle)
                + '.png', dpi=150)




def sq_distances(X, Y=None):
    """
    If Y=None, then this computes the distance between X and itself
    """
    assert (X.ndim == 2)

    # IMPLEMENT: compute pairwise distance matrix. Don't use explicit loops, but the above scipy functions
    # if X=Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert (Y.ndim == 2)
        assert (X.shape[1] == Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists


def gauss_kernel(X, Y=None, sigma=1.0):
    """
    Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they are replaced by X

    returns: kernel matrix
    """

    # IMPLEMENT: compute squared distances and kernel matrix
    sq_dists = sq_distances(X, Y)
    K = np.exp(-sq_dists / (2 * sigma ** 2))
    return K


# IMPLEMENT
def linear_kernel(X, Y):
    return np.dot(X, Y.T)


def quadratic_time_mmd(X, Y, kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X, X)
    K_XY = kernel(X, Y)
    K_YY = kernel(Y, Y)

    n = len(K_XX)
    m = len(K_YY)

    # IMPLEMENT: unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY) / (n * m)
    return mmd