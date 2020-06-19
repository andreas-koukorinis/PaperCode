from __future__ import division, print_function
from builtins import zip
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy
import scipy.stats as stats
import scipy.optimize


def z2gh(z, A, B, g, h, c=0.8):
    """
    x= A + B*(1+c*tanh(g*z/2))* z*exp(0.5*h*z^2)
    :param z:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :return:
    """
    term1 = 1 + c*np.tanh(g*z /2.0)
    term2 = z * np.exp(0.5 * h * z **2)
    if g==0:
        term1 = 1
    if h ==0:
        term2 = z
    return A + B * term1 * term2

def qgh(p, A, B, g, h, c=0.8):
    '''
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
    '''
    return z2gh(stats.norm.ppf(p), A, B, g, h, c)

def rgh(n, A, B, g, h, c=0.8):
    """
    random sampling from g, h
    :param n:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :return:
    """
    return z2gh(stats.random.rvs(size=n), A, B, g, h, c)

def pgh_scalar(q, A, B, g ,h, c=0.8, zscale = False)
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
    def _f_to_solve()

def qpk(z, loc, scale, g, h=0.01):
    '''
    transforms the normal distribution to a tukey g-h
    :param params - g:params[2] , loc:params[0] and scale:params[1]
    :param z: normal valiable
    :param h: h param from Tukey
    :return: gh transformed variable
    '''
    if np.isclose(g, 0, atol=1e-5):  # if g is close to zero then compute transformation on lim g -->0
        return loc + scale * z * np.exp(h * z ** 2.0 / 2.0)
    else:
        return loc + scale * (1.0 / g * (np.exp(g * z) - 1.0) * np.exp(h * z ** 2.0 / 2.0))


def qpkInverse(z, loc, scale, g, h):
    '''
    inverting the tukey distribution
    :param z: float
    :param loc: float
    :param scale: float
    :param g: float
    :param h: float
    :return:
    '''

    def _f(p):
        return qpk(p, loc, scale, g, h) - z

    res = scipy.optimize.root(_f, x0=[0])
    return res.x


def estimateTukeyDistribution(Y, hGrid=None):
    '''
    tukey distribution based on Sec 2.1 of Gentons paper
    :param Y: np.ndarray
    :param hGrid: (list of floats) Default. Grid search over the h parameter, and optimise for the local loc, scale, and g parameter
    :return:
    '''
    p = np.linspace(1e-2, 1e-2, 100)
    zpk = scipy.stats.norm.ppf(p)
    qhat = np.percentile(Y, p * 100)

    def qresiduals(params, zpk, qhat):
        return np.sum((qhat - qpk(zpk, *params)) ** 2)

    mu = np.mean(Y)
    std = np.maximum(np.std(Y - mu), 1e-5)

    if not hGrid is None:
        results = []
        ## Grid
        for h in hGrid:
            res = scipy.optimize.least_squares(
                qresiduals,
                x0=(mu, std, 0.8, h),
                # since h is fixed we apply tight bounds on h
                bounds=(np.array([-np.inf, 1e-6, -np.inf, h - 1e-5]), np.array([np.inf, np.inf, np.inf, h + 1e-5])),
                args=(zpk, qhat),
                max_nfev=1000,
            )

            loc, scale, g, h, fun = res.x[0], res.x[1], res.x[2], res.fun[0]
            resDict = dict(zip(["loc", "scale", "g", "h", "fun"], [loc, scale, g, h, fun]))
            results.append(resDict)
            results = pd.DataFrame(results)
        return results.sort_values("fun").iloc[0].to_dict()
    else:
        res = scipy.optimize.least_squares(
            qresiduals,
            x0=(mu, std, 0.8, 0.01),
            bounds=(np.array([-np.inf, 1e-6, -np.inf, 1e-6]), np.array([np.inf, np.inf, np.inf, np.inf])),
            args=(zpk, qhat),
            max_nfev=1000,
        )
        loc, scale, g, h, fun = res.x[0], res.x[1], res.x[2], res.x[3], res.fun[0]
        resDict = dict(zip(["loc", "scale", "g", "h", "fun"], [loc, scale, g, h, fun]))
        return resDict

        return results.sort_values("fun").iloc[0]


def rollingEstimateTukeyDistribution(Ydf, valueCol, timeCol, sampleFreq=1):
    """

    :param Ydf: pd.Dataframe
    :param valueCol: str
    :param timeCol: str
    :param sampleFreq: int
    :return:
    """
    tqdm.pandas(leave=False)
    Ydf["window"] = Ydf[timeCol].apply(lambda x: x // sampleFreq)
    return (
        Ydf.groupby("window")[[valueCol]]
            .progress_apply(lambda ts: estimateTukeyDistribution(ts[valueCol].ravel()))
            .reset_index(level=1, drop=True)
    )

    ##Functions from the R Package gk
def z2gh(z, A, B, g, h, c):
        '''
        Transform standard normal draws to g- and h- draws
        Essentially this function does this following calculation
        x= A + B * (1 + c*tanh(g*z/2)) * z * exp(0.5*h*z^2)
        :param z: vector of N(0,1) draws
        :param A: vector of location params
        :param B: vector of scale params. must be positive
        :param g: vector of g params
        :param h: vector of h params
        :param c: vector of params. often fixed at 0.8 which is the default
        :return: a vector of g- and h- values
        '''
        term1 = 1 + c * np.tanh(g * z / 2.0)
        term2 = z * np.exp(h * z ** 2.0 / 2.0)

        # must test for g=0 to return 1, andif h=0 to return z

        return A + B * term1 * term2





def pgh(q, A, B, g, h, c=0.8, zscale=False):
    """
    cumulative distribution function tukey g- and h- so it calculates the cdf F(x) given input x
    :param q: quantiles
    :param A: loc param
    :param B: scale param
    :param g: g param- parameter for skew
    :param h: h param - parameter for kurtosis
    :param c: set default to 0.8
    :param zscale: when true returns the N(0,1) quantile needed by dgh
    :return: cumulative probability
    """

    def _f(p):
        return z2gh(p, A, B, g, h, c) - q

    res = scipy.optimize.root_scalar(_f)

    if zscale:
        return res.root
    else:
        return stats.norm.cdf(res.root)

def qghDeriv(z, A, B, g, h, c=0.8, getR=False):
    """
    this function calculates the derivative of the g and h Q function.
    so calculates B*exp(h*z^2/2)*((1 + c * tanh(g*z))*(1 + h*z^2) + c*g*z/(2*cosh(g*z/2)^2))
    :param z:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :param getR: when \code{TRUE} returns a faster calculation that has the same sign as the derivative (used in checking)
    :return:
    """
    zSquared = z ** 2
    if not getR:
        term1 = np.exp(h * zSquared / 2)
    term2 = 1 + c * np.tanh(g * z / 2)
    term3 = 1 + h * zSquared
    term4 = c * g * z / (2 * np.cosh(g * z / 2) ** 2)

    if getR:
        return term2 * term3 + term4
    else:
        return B * term1 * (term2 * term3 + term4)

def qghLogDeriv(z, A, B, g, h, c=0.8):
    """

    :param z:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :return:
    """
    zSquared = z ** 2
    term1 = h * zSquared / 2.0
    term2 = 1 + c * np.tanh(g * z / 2.0)
    term3 = 1 + h * zSquared
    term4 = c * g * z / (2 * np.cosh(g * z / 2) ** 2)
    return np.log(B) + term1 + np.log(term2 * term3 + term4)

def dgh(x, A, B, g, h, c=0.8, log=False):
    """
    PDF of the distribution
    :param x:
    :param A:
    :param B:
    :param g:
    :param h:
    :param c:
    :param log:
    :return:
    """
    z = pgh(x, A, B, g, h, c=0.8, zscale=True)
    if log:
        return stats.norm.logpdf(z) - qghLogDeriv(z, A, B, g, h, c)
    else:
        return stats.norm.pdf(z) / qghLogDeriv(z, A, B, g, h, c)

class _ABCStatsMethod():
    #Approximate Baysian Computation Inference
    """
    convert octiles to estimates of location and scale, skewness and kurtosis
    octiles = vector of octiles
    """
    ALL_STATS = "all"
    OCTILES = "octiles"
    MOM_ESTIMATES = "mom_estimates"

    def orderstats(n, stats):
        """
        sample a subset of order statistics
        :param n: total number of independent draws
        :param stats:  which order stands to generate
        :return: a vector of order statistics equal in length to \code{orderstats}
        """
        p = len(stats)
        kk = np.array([0] + list(stats) + [n + 1])

        shape = kk[-1] - kk[:-1]
        w=stats.gamma.rvs(shape, size=(1,))
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
            (octiles[5]+ octiles[1] - 2*octiles[3])/momb,
            (octiles[6] - octiles[4] + octiles[2] -octiles[0]) /momb,
        ]
    def abc_batch(sobs, priorSims, simStats, M, v=None):
        """

        :param sobs:
        :param priorSims:
        :param simStats:
        :param M:
        :param v:
        :return:
        """
        summaries = np.arange([simStats(priorSims) for priorSim in priorSims])
        q = summaries.shape[1]
        var = np.var(summaries, axis=1)
        d = np.sum((summaries - np.expand_dims(sobs, axis=0))**2 / np.expand_dims(var, axis=0), axis=1)

        return priorSims[np.argsort(d)[:M], :]

    def estimateTukeyDistributionABC(x, N, statsType, rprior, batchSize= 500, M=500):
        """

        :param x: vector of observations
        :param N: number of iterations to perform
        :param statsType: which summary statistics to use, in the original package this is sumstats
        :param rprior: a function with one single argument, n, which returns a matrix with n rows consisting of samples from the prior
        :param batchSize:
        :param M: number of simulations to accept
        :return:
        """
        nSamples = x.shape[0]

        if statsType == _ABCStatsMethod.ALL_STATS:
            sobs = sorted(x)
            simStats = lambda theta: sorted(rgh(nSamples, A=theta[1], B=theta[2], g=theta[3], h=theta[4]))
        elif statsType == _ABCStatsMethod.OCTILES:
            indices = int(nSamples * np.arange(1, 8) // 8.)
            sobs = sorted(x)[indices]

            simStats = lambda theta: qgh(orderstats(nSamples, indices), A=theta[1], B=theta[2], g=theta[3], h=theta[4])

        elif statsType == _ABCStatsMethod.MOM_ESTIMATES:
            indices = int(nSamples * np.arange(1, 8) // 8)
            sobs = sorted(x)[indices]

            simStats = lambda theta: momEstimates(
                qgh(orderstats(nSamples, indices), A=theta[1], B=theta[2], g=theta[3], h=theta[4])
            )

        nBatches= nSamples // batchSize
        priorSims = rprior(M)
        for batchNum in range(nBatches):
            priorSims = abc_batch(sobs,priorSims, M)




