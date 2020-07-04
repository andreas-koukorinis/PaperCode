from __future__ import division, print_function
from builtins import zip
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.misc as misc
import scipy.special as special
import scipy.stats as stats
import scipy.optimize as optimize


try:
    logsumexp = misc.logsumexp
except AttributeError:
    logsumexp = special.logsumexp

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
    return z2gh(stats.norm.rvs(size=n), A, B, g, h, c)


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
    term4 = c * g * z / (2 * np.cosh(g * z / 2) ** 2)

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

    vectorised_version = np.vectorize(Qgh_deriv_scalar)
    return vectorised_version(z, A, B, g, h, c)


def improper_uniform_log_density(theta):
    """
    compute log-prior
    :param theta is a vector of parameters [A, B, g, k]
    :return:
    """
    if theta[1] < 0 or theta[3] < 0:
        return -np.inf
    else:
        return 0


def estimate_mcmc(x, N, theta0, Sigma0, get_log_prior=improper_uniform_log_density, t0=100, epsilon=1e-6, logB=False,
                  sd_const=1.0 / 4 * 2.4 ** 2):
    """
    MCMC inference method for g- and h- distributions
    Try to find the parameters [A,B,g, k] of the best fitting distribution
    based on a sequence of samples.
    typical example :
    x = rgh (10, A=3, B=1, g =2 , h = 0.5)
    output = mcmc(x, N = 1000, theta0 = np.array(x.mean(), x.std(), 0,0), Sigma0 = 0.1*np.eye(4))
    :param x: vector of observations
    :param N: Number of MCMC steps to perform.
    :param theta0:  vector of initial value of 4 params
    :param Sigma0:  MCMC proposal covariance matrix
    :param get_log_prior: a function with one argument (corresponding to a vector of 4 parameters e.g. A, B, g, k) returning the log-prior density. This should ensure the
    parameters are valid.
    t0: tuning parameter (number of initial iterations without adaptation)
    epsilon : tuning parameter (weight given to identity matrix in covariance calculation)
    logB: When True, the second parameter is log(B) rather than B
    :return: numpy array of size (N,4) with the evolution of each parameter through iterations and dict of parameters
    """

    output = np.zeros((N + 1, 4))
    if logB:

        def get_log_likelihood(theta):
            return dgh(x, theta[0], np.exp(theta[1]), theta[2], theta[3], log=True).sum()
    else:

        def get_log_likelihood(theta):
            return dgh(x, theta[0], theta[1], theta[2], theta[3], log=True).sum()

    output[0, :] = theta0
    theta = theta0
    Sigma = Sigma0
    C = np.linalg.cholesky(Sigma0).T
    theta_bar = 0 * theta0
    theta_mom2 = 0 * Sigma
    log_prior = get_log_prior(theta)
    log_likelihood = get_log_likelihood(theta)

    accepted_samples = 0

    for i in range(1, N):
        theta_bar = theta_bar * (i - 1) / i + theta / i
        theta_mom2 = theta_mom2 * (i - 1) / i + np.outer(theta, theta.T) / i

        if i > t0:
            M = sd_const * (theta_mom2 - np.outer(theta_bar, theta_bar.T) + epsilon * np.eye(4))
            C = np.linalg.cholesky(M).T

        theta_prop = theta + np.dot(C, stats.norm.rvs(size=4))
        log_prior_prop = get_log_prior(theta_prop)

        if log_prior_prop > -np.inf:
            log_likelihood_prop = get_log_likelihood(theta_prop)
            r = log_prior_prop + log_likelihood_prop - log_prior - log_likelihood

            if stats.uniform.rvs() < np.exp(r):
                accepted_samples += 1
                theta = theta_prop
                log_prior = log_prior_prop
                log_likelihood = log_likelihood_prop

        output[i + 1, :] = theta

    avgParam = 100
    loc, scale, g, h = (
        output[-avgParam:, 0].mean(),
        output[-avgParam:, 1].mean(),
        output[-avgParam:, 2].mean(),
        output[-avgParam:, 3].mean(),
    )
    resDict = dict(zip(["loc", "scale", "g", "h"], [loc, scale, g, h]))

    return output, resDict


def project(x, x_min, x_max):
    """
    make sure x is in [x_min, x_max] projects otherwise
    :param x:
    :param x_min:
    :param x_max:
    :return:
    """
    if (x > x_max).any():
        return np.minimum(x, x_max)
    elif (x < x_min).any():
        return np.maximum(x, x_min)
    else:
        return x


def estimate_fdsa(
        x,
        N,
        theta0,
        logB=False,
        batch_size=100,
        alpha=1,
        gamma=0.49,
        a0=1,
        c0=None,
        A=100,
        theta_min=np.array([-np.inf, 1e-5, -np.inf, 0]),
        theta_max=np.inf * np.ones(4),
):
    """
    FDSA inference method for g- and h- distribution
    try to find the best paramaters [A,B,g,k] of the best fitting distribution
    based on a sequence of samples
    :param x:
    :param N:
    :param theta0:
    :param logB:
    :param alpha:
    :param gamma:
    :param a0:
    :param c0:
    :param A:
    :param theta_min:
    :param theta_max:
    :return: A numpy array of size (N,5) with the evolution of each parameter and MLE estimate through iterations and dict of params.

     """
    if logB:
        theta_min[1] = -np.inf

        def get_log_densities(batch, theta):
            return dgh(batch, theta[0], np.exp(theta[1]), theta[2], theta[3], log=True)

    else:

        def get_log_densities(batch, theta):
            return dgh(batch, theta[0], theta[1], theta[2], theta[3], log=True)

    nobs = len(x)
    batch_size = min(batch_size, nobs)
    nm_ratio = nobs / batch_size
    theta = theta0
    estimates = np.zeros((N + 1, 5))
    estimates[0, :4] = theta

    if c0 is None:
        batch = np.random.choice(x, batch_size, replace=True)
        density_sample = get_log_densities(batch, theta0)
        c0 = density_sample.std(ddof=1) / np.sqrt(batch_size)
        c0 = np.minimum(c0, (theta_max - theta_min) / 2)  # element wise
    elif (c0 > (theta_max - theta_min) / 2).any():
        raise NameError("c0 too large compared to parameter constraints")

    for t in range(N - 1):
        at = a0 * (t + 1 + A) ** (-alpha)
        ct = c0 * (t + 1) ** (-gamma)
        batch = np.random.choice(x, batch_size, replace=True)
        gt = np.zeros(4)

        for i in range(4):
            delta = np.zeros(4)
            delta[i] = 1
            theta1 = project(theta + ct * delta, theta_min, theta_max)
            theta2 = project(theta - ct * delta, theta_min, theta_max)
            hatL1 = -nm_ratio * get_log_densities(batch, theta1).sum()
            hatL2 = -nm_ratio * get_log_densities(batch, theta2).sum()
            if np.isinf(hatL1) or np.isinf(hatL2):
                raise NameError("Log Likelihoods too small to calculate! Last value: " + str(theta))

            gt[i] = (hatL1 - hatL2) / (theta1[i] - theta2[i])

        theta = project(theta - at * gt, theta_min, theta_max)

        estimates[t + 2, :4] = theta

        estimates[t + 2, 4] = np.log(2) - logsumexp([hatL1, hatL2])  ## log of mean likelihood estimate

    avgParam = 10
    loc, scale, g, h, logLikelihood = (
        estimates[-avgParam:, 0].mean(),
        estimates[-avgParam:, 1].mean(),
        estimates[-avgParam:, 2].mean(),
        estimates[-avgParam:, 3].mean(),
        estimates[-avgParam:, 4].mean(),
    )
    resDict = dict(zip(["loc", "scale", "g", "h"], [loc, scale, g, h]))

    return estimates, resDict


class _ABCStatsMethod():
    # Approximate Baysian Computation Inference
    """
    convert octiles to estimates of location and scale, skewness and kurtosis
    octiles = vector of octiles
    """
    ALL_STATS = "all"
    OCTILES = "octiles"
    MOM_ESTIMATES = "mom_estimates"


def orderstats(n, indices):
    """
    sample a subset of order statistics
    :param n: total number of independent draws
    :param stats:  which order stands to generate
    :return: a vector of order statistics equal in length to \code{orderstats}
    """
    p = len(indices)
    kk = np.array([0] + list(indices) + [n + 1])

    shape = kk[1:] - kk[:-1]
    w = stats.gamma.rvs(shape, size=p + 1)
    return np.cumsum(w[:p]) / sum(w)


def momEstimates(octiles):
    """
    approx bayesian computation (ABC) inference for the g- and k- or g- and h- distribution
    this function performs approximate bayesian inference for iid data from a and g- and k- or g- and h- distribution, avoiding expensive computation
    the algorithm samples many parameter vectors from the prior and simulates corresponding data from the model
    the parameters are accepted or rejecte based on how similar the simulations are to the observed data
    similarity is measured using Euclidean distance between summary vectors of the simulations and observations.
    several summaries can be use, include complete order stats or summaries based on octiles.
    In the latter case, only the corresponding rows are accepted- parameter estimates plus a column giving the ABC distances.

    this function converts octiles to moment estimates
    :param octiles: vector of octiles
    :return: vector of moment estimates
    """
    momb = octiles[5] - octiles[1]
    return [
        octiles[3],
        momb,
        (octiles[5] + octiles[1] - 2 * octiles[3]) / momb,
        (octiles[6] - octiles[4] + octiles[2] - octiles[0]) / momb,
    ]


def abc_batch(sobs, priorSims, simStats, M, var=None):
    summaries = np.apply_along_axis(simStats, 1, priorSims).T
    if var is None:
        var = np.var(summaries, axis=1, ddof=1)

    d = np.sum((summaries - np.expand_dims(sobs, axis=1)) ** 2 / np.expand_dims(var, axis=1), axis=0)
    toacc = np.argsort(d)[:M]
    samp = np.concatenate((priorSims[toacc, :], d[toacc].reshape(-1, 1)), axis=1)

    return samp, var

def estimateABC(x, N, rprior, M, statsType, batchSize=1000, logB=False):
    """
    ABC inference method on g- and h- distributions

    :param x: vector of observations
    :param N: number of iterations to perform
    :param statsType: which summary statistics to use, in the original package this is sumstats
    :param rprior: a function with one single argument, n, which returns a matrix with n rows consisting of samples from the prior
    :param batchSize: mini batch size
    :param M: number of simulations to accept
    :return:
    typical example:
    x = rgh(10, A=3, B=1, g=2, h=0.5)
    rprior = lambda n : stats. uniform.rvs(loc, =0 , stats =10, size =(n, 4))
    estimateTukeyDistributionABC(x , N1E4, statsType = 'all, rprior = rprior, M = 100)
    """
    nSamples = x.shape[0]

    if statsType == _ABCStatsMethod.ALL_STATS:
        sobs = sorted(x)
        if logB:
            simStats = lambda theta: sorted(rgh(nSamples, A=theta[0], B=np.exp(theta[1]), g=theta[2], h=theta[3]))
        else:
            simStats = lambda theta: sorted(rgh(nSamples, A=theta[0], B=theta[1], g=theta[2], h=theta[3]))

    elif statsType == _ABCStatsMethod.OCTILES:
        indices = np.array([round(nSamples * i / 8) for i in range(1, 8)])
        x.sort()
        sobs = x[indices - 1]
        if logB:
            simStats = lambda theta: qgh(
                orderstats(nSamples, indices), A=theta[0], B=np.exp(theta[1]), g=theta[2], h=theta[3]
            )
        else:
            simStats = lambda theta: qgh(orderstats(nSamples, indices), A=theta[0], B=theta[1], g=theta[2], h=theta[3]
                                         )

    elif statsType == _ABCStatsMethod.MOM_ESTIMATES:
        indices = np.array([round(nSamples * i / 8) for i in range(1, 8)])
        x.sort()
        sobs = momEstimates(x[indices - 1])

        if logB:
            simStats = lambda theta: momEstimates(
                qgh(orderstats(nSamples, indices), A=theta[0], B=np.exp(theta[1]), g=theta[2], h=theta[3])
            )
        else:
            simStats = lambda theta: momEstimates(
                qgh(orderstats(nSamples, indices), theta[0], B=theta[1], g=theta[2], h=theta[3])
            )
    if N <= batchSize:
        samp = abc_batch(sobs, rprior(int(N)), simStats, M)[0]

    else:
        nBatches = int(N // batchSize + 1)
        last_batch_size = N % batchSize
        if last_batch_size == 0:
            last_batch_size = batchSize
        samp, var = abc_batch(sobs, rprior(batchSize), simStats, M)
        next_batch_size = batchSize

        for b in range(2, nBatches + 1):
            if b == nBatches:
                next_batch_size = last_batch_size

            next_samp = abc_batch(sobs, rprior(next_batch_size), simStats, M, var)[0]
            samp = np.concatenate((samp, next_samp))
            toacc = np.argsort(samp[:, 4])[:M]
            samp = samp[toacc, :]

    loc, scale, g, h, d = (
        samp[:, 0].mean(),
        samp[:, 1].mean(),
        samp[:, 2].mean(),
        samp[:, 3].mean(),
        samp[:, 4].mean(),
    )

    resDict = dict(zip(["loc", "scale", "g", "h"], [loc, scale, g, h]))

    return samp, resDict

def estimate_qls(Y, hGrid = None):
    """
    Tukey estimator based on sec 2.1 of Gentons Paper
    :param Y: np.ndarray
    :param hGrid: list(float) or None
    Grid search over h parameters and optimise the loc, scale, and g param returns best fit
    :return: best fit dictionary of params
    """

    p =np.linspace(1e-2, 1 - 1e-2, 100)

    zpk = stats.norm.ppf(p)
    qhat = np.percentile(Y, p * 100)

    def qresiduals(params, zpk, qhat):
        return np.sum(( qhat -z2gh(zpk, *params)) **2)

    mu = np.mean(Y)
    std = max(np.std(Y - mu), 1e-5)
    if not hGrid is None:
        results = []

        ## Grid

        for h in hGrid:
            res = optimize.least_squares(
                qresiduals,
                x0 = (mu, std, 0.8, h),
                # since h is fixed, we apply tight bounds on h
                bounds = (np.array([-np.inf, 1e-6, -np.inf, h - 1e-5]), np.array([np.inf, np.inf, np.inf, h + 1e-5])),
                args = (zpk, qhat),
                max_nfev=1000,
            )

            loc, scale, g, h, fun = res.x[0], res.x[1], res.x[2], res.x[0]
            resDict = dict(zip(["loc", "scale", "g", "h", "fun"], [loc, scale, g, h, fun]))
            results.append(resDict)

        results = pd.DataFrame(results)
        return results.sort_values("fun").iloc[0].to_dict()
    else:
        res = optimize.least_squares(
            qresiduals,
            x0 =(mu, std, 0.8, 0.01),
            bounds = (np.array([-np.inf, 1e-6, -np.inf, 1e-6]), np.array([np.inf, np.inf, np.inf, np.inf])),
            args = (zpk, qhat),
            max_nfev = 1000,
        )
        loc, scale, g, h, fun = res.x[0], res.x[1], res.x[2], res.x[3], res.fun[0]
        resDict = dict(zip(["loc", "scale","g","h"], [loc, scale, g, h]))
        return resDict

def estimateTukeyDistribution(Y, hGrid=None):
    """
    keeping consistency with the old code
    :param Y:
    :param hGrid:
    :return:
    """
    return estimate_qls(Y, hGrid)

def qpk(z, loc, scale, g, h):
    '''
    transforms the normal distribution to a tukey g-h
    :param params - g:params[2] , loc:params[0] and scale:params[1]
    :param z: normal valiable np.ndarray, which is a samples from a normal distribution
    :param h: h param from Tukey
    :return: gh transformed variable
    '''
    if np.isclose(g, 0, atol=1e-5):  # if g is close to zero then compute transformation on lim g -->0
        return loc + scale * z * np.exp(h * z ** 2.0 / 2.0)
    else:
        return loc + scale * (1.0 / g * (np.exp(g * z) - 1.0) * np.exp(h * z ** 2.0 / 2.0))

def qpkInverse(z, loc, scale, g, h):
    '''
    inverting the tukey distribution- so we can use this to transform the samples from a tukey distribution to a normal distribution
    :param z: float
    :param loc: float
    :param scale: float
    :param g: float
    :param h: float
    :return:
    '''

    def _f(p):
        return qpk(p, loc, scale, g, h) - z

    res = optimize.root(_f, x0=[0])
    return res.x












