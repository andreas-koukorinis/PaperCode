import os
import pandas as pd
import pickle
import numpy as np
from shogun import *
import numpy as np
import matplotlib.pyplot as plt

import random

import time
import ksd
import mmd
import shogunMMDutils as mmdutils

'''
main piece of code to do sample comparison using Shogun

'''


def do_test_compute_values():
    # this is not necessary as bootstrapping is the default
    mmd.set_null_approximation_method(NAM_PERMUTATION)
    mmd.set_statistic_type(ST_BIASED_FULL)

    # to reduce runtime, should be larger practice
    mmd.set_num_null_samples(100)


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


if __name__ == '__main__':
    print('unit testing')

    mu = 0.0
    sigma2 = 1
    b = np.sqrt(0.5)
    n = 22000
    X, Y = mmdutils.sample_gaussian_vs_laplace(n, mu, sigma2, b)
    print("Gaussian vs. Laplace")
    print("Sample means: %.2f vs %.2f" % (np.mean(X), np.mean(Y)))
    print("Samples variances: %.2f vs %.2f" % (np.var(X), np.var(Y)))

    def shogun_features(x, y):
        # create shogun features
        return [RealFeatures(x.reshape(1, len(x))) , RealFeatures(y.reshape(1, len(y)))]

    feat_p,feat_q = shogun_features(x=X, y= Y)

    kernel_width = 1
    kernel = GaussianKernel(10, kernel_width)
    kernel = GaussianKernel(10, 1)

    mmd = QuadraticTimeMMD(feat_p, feat_q)
    mmd.set_kernel(kernel)

    SignificanceResultsMMD.compute_the_p_value_unbiased(mmd=mmd)
    SignificanceResultsMMD.compute_the_p_value_biased(mmd=mmd)

    # print("%d x MMD_b[X,Y]^2=%.2f" % (len(x), biased_statistic))
    # print("%d x MMD_u[X,Y]^2=%.2f" % (len(y), unbiased_statistic))
