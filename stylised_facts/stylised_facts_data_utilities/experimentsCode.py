import os
import createLOB
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
    for symbolIDi in range(0,5):
        # iterate across 4 symbols
        for fileIDi in range(0,40):
            FileDict = {}    # create a hash like dictionary where you can basically store all the various clocks
            atocorlDict = {}
            dfInput = createLOB.rawLOBFIle(futuresFolder=extHdFutures, symbolsFolder=bmrg_folders, symbolID=symbolIDi,fileID=fileIDi)
            listDatesFiles = os.listdir("/".join((extHdFutures, bmrg_folders[symbolIDi])))
            dfCreatedLOB = createLOB.formatLOB(createLOB.createLOB(dfInput ))

            dfCreatedLOB ['DollarVolumeTraded'] = np.array(dfCreatedLOB .TradePrice)*np.array(dfCreatedLOB .TradeSize)
            clocks = createLOB.Clocks(dfCreatedLOB, 'MicroPrice', 'TradeSize', 'DollarVolumeTraded', 5, 3, 5)
            FileDict['OriginalDF'] = clocks.original_df()
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
            for counter, dataFrame in enumerate([clocks.tick_bar_df(), clocks.volume_bar_df(), clocks.dollar_bar_df()]):
                clocksEMA={}
                ticks_adj = createLOB.autoCorrel(dataFrame).ticks()
                clocksEMA[counter] =ticks_adj.ewm(span=20).mean()
                TickAtoCorrelLoc = os.path.join(folder, "".join((str(counter)+'AtoCorrelData_',string)))
                EMALoc = os.path.join(folder, "".join((str(counter)+'EMAData_',string)))
                correlarray = createLOB.autoCorrel(dataFrame).figAutocorrel(title = str(counter)+'autocorrelsymbol'+str(fileIDi),xlineLevel=80)
                pickle.dump(correlarray, open(TickAtoCorrelLoc , 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(clocksEMA, open(EMALoc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)







