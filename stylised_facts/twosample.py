import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
import seaborn as sns


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


class Kernels():
    pass


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
    # simple numpy linear kernel implementation
    return np.dot(X, Y.T)


def plot_permutation_samples_sns(null_samples, statistic=None):
    sns.distplot(null_samples)
    plt.axvline(x=np.percentile(null_samples, 2.5), ls='--', c='b')
    legend = ["95% quantiles"]
    if statistic is not None:
        plt.axvline(x=statistic, c='r')
        legend += ["Actual test statistic"]
    plt.legend(legend)
    plt.axvline(x=np.percentile(null_samples, 97.5), ls='--', c='b')
    plt.xlabel("Test statistic value")
    plt.ylabel("Counts")


def plot_permutation_samples(null_samples, statistic=None):
    plt.hist(null_samples, bins=50)
    plt.axvline(x=np.percentile(null_samples, 2.5), c='b')
    legend = ["95% quantiles"]
    if statistic is not None:
        plt.axvline(x=statistic, ls='--', c='r')
        legend += ["Actual test statistic"]
    plt.legend(legend)
    plt.axvline(x=np.percentile(null_samples, 97.5), c='b')
    plt.xlabel("Test statistic value")
    plt.ylabel("Counts")


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

