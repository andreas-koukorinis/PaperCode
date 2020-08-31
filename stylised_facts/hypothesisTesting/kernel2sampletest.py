from __future__ import division
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels

class TwoSampleKernelPython():
    # TODO: add dictionaries and or arrays so we can properly record all the data here for Latex
    def __init__(object):
        pass
    @staticmethod
    def MMD2u(K, m, n):
        """The MMD^2_u unbiased statistic.
        """
        Kx = K[:m, :m]
        Ky = K[m:, m:]
        Kxy = K[:m, m:]
        return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
               1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
               2.0 / (m * n) * Kxy.sum()

    def compute_null_distribution(self, K, m, n, iterations=10000, verbose=False,
                                  random_state=None, marker_interval=1000):
        """Compute the bootstrap null-distribution of MMD2u.
        """
        if type(random_state) == type(np.random.RandomState()):
            rng = random_state
        else:
            rng = np.random.RandomState(random_state)

        mmd2u_null = np.zeros(iterations)
        for i in range(iterations):
            if verbose and (i % marker_interval) == 0:
                print(i),
                stdout.flush()
            idx = rng.permutation(m + n)
            K_i = K[idx, idx[:, None]]
            mmd2u_null[i] = self.MMD2u(K_i, m, n)

        if verbose:
            print("")

        return mmd2u_null


    def compute_null_distribution_given_permutations(self, K, m, n, permutation,
                                                     iterations=None):
        """Compute the bootstrap null-distribution of MMD2u given
        predefined permutations.

        Note:: verbosity is removed to improve speed.
        """
        if iterations is None:
            iterations = len(permutation)

        mmd2u_null = np.zeros(iterations)
        for i in range(iterations):
            idx = permutation[i]
            K_i = K[idx, idx[:, None]]
            mmd2u_null[i] = self.MMD2u(K_i, m, n)
        return mmd2u_null

      def kernel_two_sample_test(self, X, Y, kernel_function='rbf', iterations=10000,
                               verbose=False, random_state=None, **kwargs):
        """Compute MMD^2_u, its null distribution and the p-value of the
        kernel two-sample test.

        Note that extra parameters captured by **kwargs will be passed to
        pairwise_kernels() as kernel parameters. E.g. if
        kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
        then this will result in getting the kernel through
        kernel_function(metric='rbf', gamma=0.1).
        """
        m = len(X)
        n = len(Y)
        XY = np.vstack([X, Y])
        K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
        mmd2u = self.MMD2u(K, m, n)
        if verbose:
            print("MMD^2_u = %s" % mmd2u)
            print("Computing the null distribution.")

        mmd2u_null = self.compute_null_distribution(K, m, n, iterations,
                                               verbose=verbose,
                                               random_state=random_state)
        p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() /
                      float(iterations))
        if verbose:
            print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

         return mmd2u, mmd2u_null, p_value