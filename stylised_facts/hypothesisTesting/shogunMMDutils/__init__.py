from shogun import *
import scipy
from scipy.stats import norm, laplace
import numpy as np
import matplotlib.pyplot as plt


# examples. will need refactoring


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
    plt.axvline(x=quantile, ymin=0, ymax=100, color='red',
                label=str(int(round((1 - alpha) * 100))) + '% quantile of null')
    _ = plt.legend()


class SignificanceResultsMMD(object):
    # TODO: add dictionaries and or arrays so we can properly record all the data here for Latex
    def __init__(object):
        pass

    @staticmethod
    def compute_the_p_value_unbiased(mmd):
        # now show a couple of ways to compute the test
        # compute biased and unbiased test statistic (default is unbiased)
        mmd.set_statistic_type(ST_UNBIASED_FULL)
        unbiased_statistic = mmd.compute_statistic()

        # compute p-value for computed test statistic
        p_value = mmd.compute_p_value(unbiased_statistic)
        print("P-value of MMD value for the unbiased statistic %.2f is %.2f" % (unbiased_statistic, p_value))
        return unbiased_statistic, p_value

    @staticmethod
    def compute_the_p_value_biased(mmd):
        # now show a couple of ways to compute the test
        # compute biased and unbiased test statistic (default is unbiased)
        # returns statistic and p-value
        mmd.set_statistic_type(ST_BIASED_FULL)
        biased_statistic = mmd.compute_statistic()

        # compute p-value for computed test statistic
        p_value = mmd.compute_p_value(biased_statistic)
        print("P-value of MMD value for the biased statistic %.2f is %.2f" % (biased_statistic, p_value))
        return biased_statistic, p_value

    @staticmethod
    def test_by_hand(mmd, p_value, alpha=0.05):
        # compute threshold for rejecting H_0 for a given test power
        mmd.set_statistic_type(ST_UNBIASED_FULL)
        unbiased_statistic = mmd.compute_statistic()
        threshold = mmd.compute_threshold(alpha)
        print("Threshold for rejecting H0 with a test power of %.2f is %.2f" % (alpha, threshold))

        # performing the test by hand given the above results, note that those two are equivalent
        if unbiased_statistic > threshold:
            print("H0 is rejected with confidence %.2f" % alpha)

        if p_value < alpha:
            print("H0 is rejected with confidence %.2f" % alpha)

    @staticmethod
    def full_two_sample(mmd, alpha):
        # or, compute the full two-sample test directly
        # fixed test power, binary decision
        binary_test_result = mmd.perform_test(alpha)
        if binary_test_result:
            print("H0 is rejected with confidence %.2f" % alpha)

        significance_test_result = mmd.perform_test(alpha)
        print("P-value of MMD test is %.2f" % significance_test_result)
        if significance_test_result < alpha:
            print("H0 is rejected with confidence %.2f" % alpha)

    @staticmethod
    def quadratic_time_mmd_statistics(x, y, kernelchoice):
        """
        x: data sample from variable 1
        y: data sample from variable 2
        ideally you want these two variables to be the same
        kernelChoice: pre-Set kernel in Shogun
        """
        # turn data into Shogun representation (columns vectors)
        feat_p = RealFeatures(x.reshape(1, len(x)))
        feat_q = RealFeatures(y.reshape(1, len(y)))
        # create mmd instance of test-statistic
        quad_time_mmd = QuadraticTimeMMD(feat_p, feat_q)
        quad_time_mmd.set_kernel(kernelchoice)

        # compute biased and unbiased test statistic (default is unbiased)
        quad_time_mmd.set_statistic_type(ST_BIASED_FULL)
        biased_statistic = quad_time_mmd.compute_statistic()

        quad_time_mmd.set_statistic_type(ST_UNBIASED_FULL)
        unbiased_statistic = quad_time_mmd.compute_statistic()

        return [biased_statistic, unbiased_statistic]


