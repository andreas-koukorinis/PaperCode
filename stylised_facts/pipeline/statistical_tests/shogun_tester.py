import numpy as np
import shogun as sg

from matplotlib import pyplot as plt
from scipy.stats import laplace, norm


class ShogunTester:
    # TODO: Need to get this into a runnable class instead of working example.
    #  Need some clarity on how its going to be used
    def __init__(self, n, mu, sigma_squared, b):
        self._n = n
        self._mu = mu
        self._sigma_squared = sigma_squared
        self._b = b

        self._x = None
        self._y = None

        self._statistic = None
        self._mmd = None

    def sample_gaussian_vs_laplace(self):
        self._x = norm.rvs(size=self._n) * np.sqrt(self._sigma_squared) + self._mu
        self._y = laplace.rvs(size=self._n, loc=self._mu, scale=self._b)
        return self

    def get_shogun_statistics(self):
        # turn data into Shogun representation (columns vectors)
        feat_p = sg.RealFeatures(self._x.reshape(1, len(self._x)))
        feat_q = sg.RealFeatures(self._y.reshape(1, len(self._y)))

        # choose kernel for testing. Here: Gaussian
        kernel_width = 1
        kernel = sg.GaussianKernel(10, kernel_width)

        # create mmd instance of test-statistic
        self._mmd = sg.QuadraticTimeMMD()
        self._mmd.set_kernel(kernel)
        self._mmd.set_p(feat_p)
        self._mmd.set_q(feat_q)

        # compute biased and unbiased test statistic (default is unbiased)
        self._mmd.set_statistic_type(sg.ST_BIASED_FULL)
        biased_statistic = self._mmd.compute_statistic()

        self._mmd.set_statistic_type(sg.ST_UNBIASED_FULL)
        unbiased_statistic = self._mmd.compute_statistic()

        self._statistic = unbiased_statistic

        print("\nShogun tests statistics:")
        print(f"biased test statistic {len(self._x)} x MMD_b[X,Y]^2={biased_statistic:.2f}")
        print(f"unbiased test statistic {len(self._x)} x MMD_u[X,Y]^2={unbiased_statistic:.2f}")
        return self

    def hypothesis_test_using_shogun(self, alpha=0.05):
        print("\nTesting H0:\n")
        self._mmd.set_null_approximation_method(sg.NAM_PERMUTATION)
        self._mmd.set_num_null_samples(self._n)

        # now show a couple of ways to compute the test

        # compute p-value for computed test statistic
        p_value = self._mmd.compute_p_value(self._statistic)
        print(f"p-value is {p_value:.6f} for MMD statistic of {self._statistic:.2f}")

        # compute threshold for rejecting H_0 for a given test power
        threshold = self._mmd.compute_threshold(alpha)
        print(f"Threshold for rejecting H0 with a test power of {alpha:.2f} is {threshold:.2f}")

        # performing the test by hand given the above results, note that those two are equivalent
        if self._statistic > threshold:
            print(f"H0 is rejected with a confidence level of {1-alpha:.2f}")

        if p_value < alpha:
            print(f"H0 is rejected with a confidence level of {1-alpha:.2f}")

        # or, compute the full two-sample test directly
        # fixed test power, binary decision
        binary_test_result = self._mmd.perform_test(alpha)
        if binary_test_result:
            print(f"H0 is rejected with a confidence level of {1-alpha:.2f}")
        return self

    def plot_distributions(self):
        plt.figure(figsize=(18, 5))
        plt.suptitle("Gaussian vs. Laplace")
        plt.subplot(121)
        xs = np.linspace(-2, 2, 500)
        plt.plot(xs, norm.pdf(xs, loc=self._mu, scale=self._sigma_squared))
        plt.plot(xs, laplace.pdf(xs, loc=self._mu, scale=self._b))
        plt.title("Densities")
        plt.xlabel("$x$")
        plt.ylabel("$p(x)$")

        plt.subplot(122)
        plt.hist(self._x, alpha=0.5)
        plt.xlim([-5, 5])
        plt.ylim([0, self._n*0.65])
        plt.hist(self._y, alpha=0.5)
        plt.xlim([-5, 5])
        plt.ylim([0, self._n*0.65])
        plt.legend(["Gaussian", "Laplace"])
        plt.title('Samples')

        print('Gaussian vs. Laplace')
        print(f"Sample means: {np.mean(self._x):.2f} vs {np.mean(self._y):.2f}")
        print(f"Samples variances: {np.var(self._x):.2f} vs {np.var(self._y):.2f}")

        plt.show()
        return self

    def visualise_distribution_test_statistic(self, alpha=0.05):
        num_samples = 500

        # we first sample null distribution
        null_samples = self._mmd.sample_null()

        # we then sample alternative distribution, generate new data for that
        alt_samples = np.zeros(num_samples)
        for i in range(num_samples):
            x = norm.rvs(size=self._n, loc=self._mu, scale=self._sigma_squared)
            y = laplace.rvs(size=self._n, loc=self._mu, scale=self._b)
            feat_p = sg.RealFeatures(np.reshape(x, (1, len(x))))
            feat_q = sg.RealFeatures(np.reshape(y, (1, len(y))))

            kernel_width = 1
            kernel = sg.GaussianKernel(10, kernel_width)

            mmd = sg.QuadraticTimeMMD()
            mmd.set_kernel(kernel)

            mmd.set_p(feat_p)
            mmd.set_q(feat_q)
            alt_samples[i] = mmd.compute_statistic()

        np.std(alt_samples)

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
        quantile_idx = int(len(null_samples) * (1 - alpha))
        quantile = null_samples_sorted[quantile_idx]
        plt.axvline(x=quantile, ymin=0, ymax=100, color='red',
                    label=str(int(round((1 - alpha) * 100))) + '% quantile of null')
        plt.show()
        return self


if __name__ == '__main__':
    tester = ShogunTester(1000, 0, 1, np.sqrt(0.5))
    tester.\
        sample_gaussian_vs_laplace(). \
        plot_distributions(). \
        get_shogun_statistics(). \
        hypothesis_test_using_shogun(alpha=0.05). \
        visualise_distribution_test_statistic(alpha=0.05)

