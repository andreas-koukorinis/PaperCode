from __future__ import division, print_function
from builtins import zip
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.misc as misc
import scipy.stats as stats
import scipy.optimize as optimize


def z2gh(z, A, B, g, h, c=0.8):
    """
    Transformation of the standard normal distribution
    x= A + B*(1+c*tanh(g*z/2))* z*exp(0.5*h*z^2)
    :param z:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :return:
    """
    term1 = 1 + c * np.tanh(g * z / 2.0)
    term2 = z * np.exp(0.5 * h * z ** 2)
    if g == 0:
        term1 = 1
    if h == 0:
        term2 = z
    return A + B * term1 * term2


def qgh(p, A, B, g, h, c=0.8):
    """
    g- and h- params
    Compute the quantile function
    uses stats.norm.ppf which is the percent point function - inversion of CDF and returns percentiles
    :param p: vector of probabilities
    :param A: vector of loc params
    :param B: vector of scale params. must be positive
    :param g: vector of g params
    :param h: vector of h params
    :param c: default is 0.8
    :return:
    """
    return z2gh(stats.norm.ppf(p), A, B, g, h, c)


def rgh(n, A, B, g, h, c=0.8):
    """
    computer random sampling from g, h
    :param n:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :return:
    """
    return z2gh(stats.random.rvs(size=n), A, B, g, h, c)


# noinspection PyBroadException
def pgh_scalar(q, A, B, g, h, c=0.8, zscale=False):
    """
    compute the cumulative distribution function 
    :param q: 
    :param A: 
    :param B: 
    :param g: 
    :param h: 
    :param c: 
    :param zscale: 
    :return: 
    """

    def _f_to_solve(z):
        return z2gh(z, A, B, g, h) - q

    try:
        z_sol = optimize.brentq(_f_to_solve, -100, 100)
    except:
        z_sol = np.inf * (q - A)
    if zscale:
        return z_sol
    return stats.norm.cdf(z_sol)


def pgh(q, A, B, g, h, c=0.8, zscale=False):
    """
    Vectorised version of pgh_scalar
    Compute the cum distribution function
    :param q:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :param zscale:
    :return:
    """
    vectorised_function = np.vectorize(pgh_scalar)
    return vectorised_function(q, A, B, g, h, c, zscale)


def dgh(x, A, B, g, h, c=0.8, log=False):
    """
    PDF of the distribution of f(x) = I(z)/Q'(z)

    """
    z = pgh(x, A, B, g, h, c=0.8, zscale=True)
    if log:
        return stats.norm.logpdf(z) - Qgh_log_deriv(z, A, B, g, h, c)
    else:
        return stats.norm.pdf(z) / Qgh_deriv(z, A, B, g, h, c)


def Qgh_log_deriv_scalar(z, A, B, g, h, c):
    """
    compute log derivative of Qgh
    return log(B) + (h*z^2)/2 + log((1+c*tanh(g*z/2))*(1+h*z ^2) + c*g*z/2(2*cosh(g*z/2)^2))
    """
    z_squared = z ** 2
    term1 = h * z_squared / 2
    term2 = 1 + c * np.tanh(g * z / 2)
    term3 = 1 + h * z_squared
    term4 = c * g * z / (2 * np.cosh(g * z / 2) ** 2)

    if g == 0:
        term2 = 1
        term4 = 0
    if np.isinf(z):
        term4 = 0

    return np.log(B) + term1 + np.log(term2 * term3 + term4)


def Qgh_log_deriv(z, A, B, g, h, c=0.8):
    """
    Vectorised version of the Qgh_log_deriv
    """
    vectorised_version = np.vectorize(Qgh_log_deriv_scalar)

    return vectorised_version(z, A, B, g, h, c)


def Qgh_deriv_scalar(z, A, B, g, h, c):
    z_squared = z ** 2
    term1 = np.exp(h * z_squared / 2)
    term2 = 1 + c * np.tanh(g * z / 2)
    term3 = 1 + h * z_squared
    term4 = c * g * z / (2 * np.cosh(g * z/2) ** 2)

    if g == 0:
        term2 = 1
        term4 = 0
    if np.isinf(z):
        term4 = 0

    return B * term1 * (term2 * term3 + term4)

def Qgh_deriv(z, A, B, g, h, c=0.8):
    """
    vectorised version of the Qgh_deriv_scalar
    :param z:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :return:
    """

    vectorised_version = np.vectorize( Qgh_deriv_scalar)
    return vectorised_version(z, A, B, g, h ,c)
