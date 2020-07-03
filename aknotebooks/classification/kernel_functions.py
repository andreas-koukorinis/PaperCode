import numpy as np
from numpy import linalg


def map_func(x):
    """
    Map d-dimensional data to d+1-dimensions.  Return a numpy array of data points.

    x : matrix, a collection of data points, shape = [n_samples, n_features]
    """
    if len(x.shape) == 1:
        # x contains only a single data point
        return np.r_[x, calc_z(x)]
    else:
        z = [np.r_[j, calc_z(j)] for j in x]
        return np.array(z)


def calc_z(x):
    """
    Take the dot product of an array with itself.

    x : array, shape = [1, n_features]
    """
    return np.dot(x, x.T)


def my_kernel(x1, x2):
    """
    Custom kernel to pass to sklearn svm.SVC via its kernel parameter

    x1 : matrix, a collection of data points, shape = [n_samples, n_features]
    x2 : matrix, a collection of data points, shape = [n_samples, n_features]
    """
    return np.dot(map_func(x1), map_func(x2).T)


def calc_plane_norm(sv, coef):
    """
    Calculate the normal to the hyperplane (in mapped space)

    sv : matrix, contains mapped points, shape = [n_supportvectors, n_mappedfeatures]
    coef : array of floats, shape = [n_supportvectors, 1]
    """
    components = coef[:, np.newaxis] * sv
    return np.sum(components, axis=0)


def calc_z_plane(x):
    
    """
    Calculate z-coordinates of the decision plane

    x: matrix, shape = [n_samples, n_features]
    """
    return (-w[0] * x[0] - w[1] * x[1] - b) / w[2]


class Kernel(object):
    @staticmethod
    def linear():
        def f(x, y):
            return np.dot(x, y.T)

        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            return np.exp(-linalg.norm(x - y.T) ** 2 / (2 * (sigma ** 2)))

        return f

    @staticmethod
    def polynomial(dim, offset):
        def f(x, y):
            return (offset + np.dot(x, y.T)) ** dim

        return f

    @staticmethod
    def inhomogenous_polynomial(dim, offset):
        return kernel.polynomial(dim=dim, offset=1.0)

    @staticmethod
    def homogenous_polynomial(dim, offset):
        return kernel.polynomial(dim=dim, offset=0.0)

    @staticmethod
    def tanh(kappa, c):
        def f(x, y):
            return numpy.tanh(kappa * numpy.dot(x, y) + c)

        return f
