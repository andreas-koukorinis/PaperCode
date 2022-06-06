import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import lob_for_futures as lobFut
from lob_for_futures import *
import os
import matplotlib.pyplot as plt

from dateutil.parser import parse
from pandas.tseries.offsets import BDay
import pickle as pkl
import fnmatch
import pickle
import datetime as dt
from datetime import timedelta
import glob
import fathon
from fathon import fathonUtils as fu
import itertools
import pyinform as pyinf
import pingouin as pig
import time
import matplotlib.ticker as mtick

dataFolder = lobFut.dataFolder
quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]
elements = '/media/ak/Elements/OrderBookProcessedData'

procesSymbols = sorted(os.listdir(elements))
symbolIdx = 0

symbols = [f.split("_")[0] for f in quotes]
#
symbol = sorted(symbols)[symbolIdx]
print(symbol)
quotesFileCh = os.path.join(dataFolder, quotes[symbolIdx])
tradesFileCh = os.path.join(dataFolder, trades[symbolIdx])
#
# # get common Dates
quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])
tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])
intersectionDates = list(set(quotesDates).intersection(tradesDates))

trades_cols = ['size', 'time', 'type', 'value']
# experimentsDestination = '/media/ak/T7/MFDFA Experiments'
plt.style.use(os.path.join('/home/ak/.config/matplotlib', 'latexstyle3.mplstyle'))
if __name__ == '__main__':
    pass
