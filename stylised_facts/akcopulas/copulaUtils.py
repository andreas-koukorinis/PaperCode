import math
import numpy as np
import scipy as sp
import bokeh as bk
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import xlwt
from bokeh.plotting import figure, output_file, show


def tCopula_3D_TVP_LL(theta, U, exog12, exog13, exog23, exogNU):

   """
   Returns -ve LL of the 3-dimension Student R copula

   :param theta:
   :param U:
   :param exog12:
   :param exog13:
   :param exog23:
   :param exogNU:
   :return:
   """

    T, N = U.shape

    k12 = exog12.shape[1]
    k13 = exog13.shape[1]
    k23 = exog23.shape[1]
    kNU = exogNU.shape[1]

    beta12 = theta[:k12]
    beta13 = theta[k12:k12 + k13]
    beta23 = theta[k12 + k13:k12 + k13 + k23]
    betaNU = theta[k12 + k13 + k23:]

    NUt = 2 + np.exp(np.dot(exogNU, betaNU))
    rho12 = -1 + 2 / (1 + np.exp(np.dot(-exog12, beta12)))
    rho13 = -1 + 2 / (1 + np.exp(np.dot(-exog13, beta13)))
    rho23 = -1 + 2 / (1 + np.exp(np.dot(-exog23, beta23)))

    RHOt = np.empty((3, 3, T))
    counter = 0
    for tt in np.arange(T):
        #        allRHOt = np.concatenate((rho12[:tt],rho13[:tt],rho23[:tt]),axis=0)
        allRHOt = np.vstack((rho12[tt], rho13[tt], rho23[tt]))
        RHOt[:, :, tt] = theta2rho(allRHOt)
        if is_pos_def(RHOt[:, :, tt]) == 0:
            counter = counter + 1;

    if counter > 0:
        LL = 1e7
    else:
        trmU = np.empty((T, N))
        LLa = np.empty((T, 1))

        for tt in np.arange(T):
            NU = NUt[tt]

            if NU < 100:
                trmU[tt, :] = stats.t.ppf(U[tt, :], NU)
            else:
                trmU[tt, :] = stats.norm.ppf(U[tt, :])

            allRHOt = np.vstack((rho12[tt], rho13[tt], rho23[tt]))
            RHO = theta2rho(allRHOt)

            LLa[tt] = math.lgamma((NU + N) / 2) + (N - 1) * math.lgamma(NU / 2) - N * math.lgamma(
                (NU + 1) / 2) - 0.5 * math.log(np.linalg.det(RHO))
            t1 = trmU[tt, :].dot(np.linalg.inv(RHO)).dot(trmU[tt, :])
            # np.reduce(np.dot,[trmU[tt,:],np.linalg.inv(RHO),trmU[tt,:]])
            # Equiv. to t0 =
            LLa[tt] = LLa[tt] - ((NU + N) / 2) * np.log(1 + t1 / NU)
            LLa[tt] = LLa[tt] + ((NU + 1) / 2) * sum(np.log(1 + (trmU[tt, :] ** 2 / NU)))
        LL = -sum(LLa)
        print(LL)
        return LL
        # return LL,rho12,rho13,rho23,NUt

def multivariatet(mu,Sigma,df,N):

   dim = len(Sigma)
   g = np.tile(np.random.gamma(df/2.,2./df,M),(dim,1)).T
   Z = np.random.multivariate_normal(np.zeros(dim),Sigma,N)
   return mu + Z/np.sqrt(g)