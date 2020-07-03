import os
import numpy as np
import pickle as pkl
import  reateLOB
from collections import defaultdict

folder= '/media/ak/My Passport/Experiment Data/ActivityClockData/'
folderList = os.listdir(folder)
folderList
clocksDataList =[s for s in folderList if ('ClocksData') in s]
# correlDataList =[s for s in folderList if ('AtoCor') in s]