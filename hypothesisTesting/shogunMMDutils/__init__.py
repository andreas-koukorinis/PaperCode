from shogun import *
import scipy
from scipy.stats import norm, laplace
import numpy as np
import matplotlib.pyplot as plt
#examples. will need refactoring




def sample_gaussian_vs_laplace(n=220, mu=0.0, sigma2=1, b=np.sqrt(0.5)):
    '''
    use scipy to generate two distributions. in our case it is just laplace and normal
    '''
    X = norm.rvs(size=n, loc=mu, scale=sigma2)
    Y = laplace.rvs(size=n, loc=mu, scale=b)

    return X, Y


def plot_alt_vs_null(alt_samples=None, null_samples=None, alpha=None):
    plt.figure(figsize=(18, 5))

    plt.subplot(131)
    plt.hist(null_samples, 50, color='blue')
    plt.title('Null distribution')
    plt.subplot(132)
    plt.title('Alternative distribution')
    plt.hist(alt_samples, 50, color='green')

    plt.subplot(133)
    plt.hist(null_samples, 50, color='blue')
    plt.hist(alt_samples, 50, color='green', alpha=0.5)
    plt.title('Null and alternative distriution')

    # find (1-alpha) element of null distribution
    null_samples_sorted = np.sort(null_samples)
    quantile_idx = int(num_samples * (1 - alpha))
    quantile = null_samples_sorted[quantile_idx]
    plt.axvline(x=quantile, ymin=0, ymax=100, color='red', label=str(int(round((1 - alpha) * 100))) + '% quantile of null')
    _ = plt.legend()