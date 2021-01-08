import os
import pandas as pd
import pickle
import numpy as np
from shogun import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
import random

import time
import ksd
import mmd
import shogunMMDutils as mmdutils
from shogun import RealFeatures

'''
main piece of code to do sample comparison using Shogun

'''

if __name__ == '__main__':
    print('unit testing')

    mu = 0.0
    sigma2 = 1
    b = np.sqrt(0.5)
    n = 2200
    X, Y = mmdutils.sample_gaussian_vs_laplace(n, mu, sigma2, b)
    print("Gaussian vs. Laplace")
    print("Sample means: %.2f vs %.2f" % (np.mean(X), np.mean(Y)))
    print("Samples variances: %.2f vs %.2f" % (np.var(X), np.var(Y)))


    def shogun_features(x, y):
        # create shogun features
        return [RealFeatures(x.reshape(1, len(x))), RealFeatures(y.reshape(1, len(y)))]


    feat_p, feat_q = shogun_features(x=X, y=Y)
    mmd = QuadraticTimeMMD(feat_p, feat_q)
    kernel = GaussianKernel(10, 1)
    mmd.set_kernel(kernel)
    kernel1 = GaussianKernel(10, 0.1)
    kernel2 = GaussianKernel(10, 1)
    kernel3 = GaussianKernel(10, 10)
    kernel4 = GaussianKernel(5, 0.15)
    mmd.add_kernel(kernel1)
    mmd.add_kernel(kernel2)
    mmd.add_kernel(kernel3)
    mmd.add_kernel(kernel4)

    mmd.set_train_test_mode(True)
    mmd.set_train_test_ratio(1)

    # mmd.set_kernel_selection_strategy(KSM_MAXIMIZE_POWER)
    # mmd.select_kernel()
    # learnt_kernel_single = GaussianKernel.obtain_from_generic(mmd.get_kernel())
    # width = learnt_kernel_single.get_width()
    # print(width)
    num_runs = 1
    num_folds = 3
    alpha = 0.05
    mmd.set_kernel_selection_strategy(KSM_CROSS_VALIDATION, num_runs, num_folds, alpha)
    mmd.select_kernel()
    learnt_kernel_single = GaussianKernel.obtain_from_generic(mmd.get_kernel())
    width = learnt_kernel_single.get_width()
    print('Print best kernel width is:', width)

    mmdutils.SignificanceResultsMMD.compute_the_p_value_unbiased(mmd=mmd)
    _, p_value_biased = mmdutils.SignificanceResultsMMD.compute_the_p_value_biased(mmd=mmd)
    _, p_value_unbiased = mmdutils.SignificanceResultsMMD.compute_the_p_value_unbiased(mmd=mmd)

    mmdutils.SignificanceResultsMMD.test_by_hand(mmd=mmd, p_value=p_value_unbiased, alpha=alpha)
    mmdutils.SignificanceResultsMMD.full_two_sample(mmd=mmd, alpha=alpha)

