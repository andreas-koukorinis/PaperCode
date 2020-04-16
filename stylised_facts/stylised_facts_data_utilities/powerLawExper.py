import os
import createLOB
from collections import defaultdict
import time
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from powerlaw import plot_pdf, Fit, pdf

print('test')
#folders / fileC
folder= '/media/ak/My Passport/Experiment Data/ActivityClockData/'
folderList = os.listdir(folder)
symbols =['FB1','JB1','FV1','G_1','DU1']
mergedDFs = defaultdict(dict)
for symbol in symbols:
    start = time.time()
    ListClocksData =list(np.sort([s for s in folderList if ('ClocksData') in s and ('_'+str(symbol)) in s]))
    print(ListClocksData)
    for clocksIdx, _ in enumerate(ListClocksData):
        #fileLoc = "".join((folder, ListClocksData[clocksIdx]))
        dFClocksSample = pickle.load(open("".join((folder, ListClocksData[clocksIdx])), "rb"))
        sampleKeys = list(dFClocksSample.keys())
        column = 'MicroPrice'
        ref = dFClocksSample[sampleKeys[0]][str(column)]
        sub = dFClocksSample[sampleKeys[1]][str(column)]
        ohlcDF = createLOB.get_ohlc(ref, sub)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - start))


