import fathon
import pandas as pd
from fathon import fathonUtils as fu
import numpy as np
import os

# LOCATIONS
# laptop_OS_folder = '/media/ak/OS/Data/FuturesDataSemiProcessed'
# LaCie_ProcessedData = '/media/ak/LaCie/ProcessedSampledData/'
# # returns_data = '/media/ak/T7/August11th2022Experiments/Returns/'
t7 = '/media/ak/T7/'
# june_ext = os.path.join(t7, 'June4th2022Experiments')
# returns_data = [f for f in os.listdir(june_ext) if '_returns' in f]
mainFolder = '/media/ak/T71/August11th2022Experiments/'
experimentsLocation = os.path.join(mainFolder, 'ExperimentInputFiles')
bars = ['tick', 'volume', 'calendar', 'dollar']


def read_exp_data(symbol_Path, bar, file_idx_):
    symbol_files_ = [f for f in os.listdir(symbol_Path) if str(bar) in f]
    file_idx_loc_ = os.path.join(symbol_Path, symbol_files_[file_idx_])
    df_idx_ = pd.read_pickle(file_idx_loc_)
    return df_idx_


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


if __name__ == '__main__':
    symbol = 'JB1'
    bar = 'tick'
    symbolPath = os.path.join(experimentsLocation, symbol)
    symbol_file_idx = 100

    exp_df_idx = read_exp_data(symbolPath, bar,  symbol_file_idx)  # experiment df indexed
    data_input = to_agg(exp_df_idx.pct_change_micro_price)
    print(data_input)
    winSizes_ = fu.linRangeByStep(10, win_Sizes_len(data_input))  # rounding length of t to the nearest 100

    qs = np.arange(-3, 4, 0.1)
    revSeg = True
    polOrd = 1
    testClassOne = mfdfaquantities(data_input, winSizes_, qs, revSeg, polOrd)
    print(testClassOne.compute_mass_exponents())
