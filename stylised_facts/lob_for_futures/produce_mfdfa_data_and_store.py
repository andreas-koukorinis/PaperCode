import pandas as pd
import fathon
from fathon import fathonUtils as fu
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import os
import numpy as np

scaler = MinMaxScaler()
standard_scaler = StandardScaler()

sys.path.insert(0, '/directory/tothe/handshakefile/')
sys.path.append('/home/ak/Documents/PaperCode/stylised_facts')


def to_agg(data):
    return fu.toAggregated(data)


def win_Sizes_len(data):
    return round(len(to_agg(data)), -2)


class mfdfaquantities(object):
    """
 winSizes : numpy ndarray
            Array of window's sizes.
        qList : float or iterable or numpy ndarray
            List of q-orders used to compute `F`.
        polOrd : int, optional
            Order of the polynomial to be fitted in every window (default : 1).
        revSeg : bool, optional
            If True, the computation of `F` is repeated starting from the end of the time series (default : False).
"""

    def __init__(self, data, winSizes, qs, revSeg, polOrd):
        self.data = data
        self.winSizes = winSizes
        self.qs = qs
        self.revSeg = revSeg
        self.polOrd = polOrd
        self.pymfdfa = fathon.MFDFA(self.data)

    def n_F_output(self):
        """
        # F : numpy ndarray
           Returns
            -------
            numpy ndarray
                Array `n` of window's sizes.
            numpy ndarray
                qxn array `F` containing the values of the fluctuations in every window for every q-order.
        """
        n, F = self.pymfdfa.computeFlucVec(self.winSizes, self.qs, self.revSeg, self.polOrd)
        return n, F

    def H_and_H_intcpt_output(self):
        """
        listH : numpy ndarray
        # Array containing the values of the slope of the fit at every q-order.
        # fit of the fluctuation values
                Returns
            -------
            numpy ndarray
                Slope of the fit for every q-order.
            numpy ndarray
                Intercept of the fit for every q-order.


        """
        list_H, list_H_intercept = self.pymfdfa.fitFlucVec()
        return list_H, list_H_intercept

    def compute_mass_exponents(self):
        tau = self.pymfdfa.computeMassExponents()
        return tau

    def compute_multi_fractal_spectrum(self):
        alpha, mfSpect = self.pymfdfa.computeMultifractalSpectrum()
        return alpha, mfSpect


def read_pkl_idx(files, file_idx):
    file_idx_loc = os.path.join(experimentsLocation, files[file_idx])
    dict_idx = pd.read_pickle(file_idx_loc)
    tick_ = dict_idx['tick']
    volume_ = dict_idx['volume']
    calendar_ = dict_idx['calendar']
    dollar_ = dict_idx['dollar']
    return tick_, volume_, calendar_, dollar_


experimentsLocation = '/media/ak/T71/August11th2022Experiments/'
if __name__ == '__main__':
    rx_files = [f for f in os.listdir(experimentsLocation) if str('RX1_') in f]
    print(rx_files)
    tick, volume, calendar, dollar = read_pkl_idx(rx_files, 1)
    data = tick.pct_change_micro_price
    print(data)

    winSizes = fu.linRangeByStep(10, 2000)
    qs = np.arange(-3, 4, 0.1)
    revSeg = True
    polOrd = 1
    data_ = volume.pct_change_micro_price # chose data
    testClass = mfdfaquantities(data_, winSizes, qs, revSeg, polOrd)

    # need to figure out how to parallelise this better
    n, F = testClass.n_F_output()
    list_H, list_H_intercept = testClass.H_and_H_intcpt_output()
    tau = testClass.compute_mass_exponents()
    alpha, mfSpect = testClass.compute_multi_fractal_spectrum()

    print(alpha)