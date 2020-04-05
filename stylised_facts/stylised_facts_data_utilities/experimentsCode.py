import os
import stylised_facts.stylised_facts_data_utilities.createLOB as createLOB
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns

plt.style.use('seaborn-talk')
plt.style.use('bmh')

# plt.rcParams['font.family'] = 'DejaVu Sans Mono'
# plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'
# plt.rcParams['figure.figsize'] = 10,7
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)

# import util libs

from tqdm import tqdm, tqdm_notebook
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 777

### Files / Folders /
extHD = '/media/ak/My Passport/'
extHdData = "".join((extHD, 'Data'))
extHdExpData = "".join((extHD, 'Experiment Data'))  # ['features', 'labels', 'metrics', 'models']
extHdFutures = "".join((extHD, 'Barket Data/raw bloomberg data'))  # futures
bmrg_folders=[s for s in os.listdir(extHdFutures) if ('Comdty') in s]

# cleanLOBFolder = "/".join((extHdExpData, 'CleanLOBData'))
# symbols = sorted(os.listdir(cleanLOBFolder))

if __name__ == '__main__':
    symbolIDi = 2
    for fileIDi in range(0,10):
        FileDict = {}    # create a hash like dictionary where you can basically store all the various clocks
        dfInput = createLOB.rawLOBFIle(futuresFolder=extHdFutures, symbolsFolder=bmrg_folders, symbolID=symbolIDi,fileID=fileIDi)
        listDatesFiles = os.listdir("/".join((extHdFutures, bmrg_folders[symbolIDi])))
        dfCreatedLOB = createLOB.formatLOB(createLOB.createLOB(dfInput ))

        dfCreatedLOB ['DollarVolumeTraded'] = np.array(dfCreatedLOB .TradePrice)*np.array(dfCreatedLOB .TradeSize)
        clocks = createLOB.Clocks(dfCreatedLOB, 'MicroPrice', 'TradeSize', 'DollarVolumeTraded', 5, 3, 5)
        mad = clocks.mad_outlier(dfCreatedLOB.MicroPrice.values.reshape(-1, 1))
        FileDict['TickBarDf'] = clocks.tick_bar_df()
        FileDict['VolumeBarDf'] = clocks.volume_bar_df()
        FileDict['DollarVolumeBarDf'] = clocks.dollar_bar_df()

        # save the output of the folder ###
        folder= '/media/ak/My Passport/Experiment Data/ActivityClockData/'
        string = "_".join((listDatesFiles[fileIDi].split('-')[0], listDatesFiles[fileIDi].split('-')[1].split('.')[0],'.pkl')) #take out symbol and date

        FileName ="".join(('ClocksData_',string))

        save_loc = os.path.join(folder, FileName)
        pickle.dump(FileDict, open(save_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print('just saved: ', save_loc)






