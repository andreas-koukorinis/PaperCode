# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from shogun import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import time
import os
import pandas as pd

# import stylised_facts.stylised_facts_data_utilities.createLOB as createLOB

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    etfDir = '/media/ak/My Passport1/MarketData/ETF_Levels/'
    etfSymbols = sorted(os.listdir(etfDir))
    etfIdx = 4  # index for etf file
    print('Pick the', etfIdx, 'symbol to play with')
    ### combine symbol index with etfDir to get all the location
    etfSymbolFiles = os.path.join(etfDir, etfSymbols[etfIdx])
    dfConcat= pd.read_csv( os.path.join(etfSymbolFiles, 'microPriceConcatenated.csv'))
    microPriceArray = dfConcat.dropna().values
    mPnonZero = microPriceArray[microPriceArray != 0]
    subset =mPnonZero[0:15000]
    msk = np.random.rand(len(subset)) < 0.55
    train = subset[msk]

    test = subset[~msk]
    features_p = RealFeatures(train.reshape(1, len(train)))
    features_q = RealFeatures(test.reshape(1, len(test)))

    mmd = LinearTimeMMD()
    kernel = GaussianKernel(10, 1)
    mmd.set_kernel(kernel)
    mmd.set_p(features_p)
    mmd.set_q(features_q)
    mmd.set_num_samples_p(1000)
    mmd.set_num_samples_q(1000)
    alpha = 0.05

    mmd.set_statistic_type(ST_BIASED_FULL)
    statistic = mmd.compute_statistic()

    threshold = mmd.compute_threshold(alpha)
    p_value = mmd.compute_p_value(statistic)
    print(p_value)
