
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
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
#plt.style.use(os.path.join('/home/ak/.config/matplotlib','latexstyle.mplstyle'))
import stylised_facts_data_utilities.createLOB as createLOB

def open_pickle_file(path, pickle_file):
    file_loc = os.path.join(path, pickle_file)
    pickle_to_file = pickle.load(open(file_loc, "rb"))
    return pickle_to_file

extPath = '/media/ak/My Passport/ExperimentData'
barketData = '/media/ak/My Passport/Barket Data/'
#cleanBloombergDataLocation = os.path.join(dataOnlyDrive, 'CleanBloombergData')
destinationFolder = '/media/ak/My Passport/ExperimentData/ProcessedDataForExperiments'
bmrg_folders = [s for s in os.listdir(barketData) if ('5y') in s]
targetDrive = barketData


if __name__ == '__main__':
    Folder = os.path.join(targetDrive, bmrg_folders[0])
    # tradesFolder = os.path.join(targetDrive, bmrg_folders[1])

    list_Files = os.listdir(Folder)
    lob_dict = {}

    for idx, fileName in enumerate(list_Files):
        fileLocation = os.path.join(Folder, list_Files[idx])

        raw_file = pd.read_csv(fileLocation)
        exampleLOB = createLOB.calcLOB(createLOB.formatLOB(createLOB.createLOB(raw_file)))
        lob_dict[idx] = exampleLOB
        allLOB = pd.concat(list(lob_dict.values()), axis=0, keys=list(lob_dict.keys()))
    allLOB_File = "".join(('allLOB' + str(bmrg_folders[0])   , '.pkl'))
    pkl.dump(allLOB, open("/".join((extPath, allLOB_File)), "wb"))