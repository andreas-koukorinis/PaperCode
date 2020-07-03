import os
import numpy as np
import pickle as pkl
import createLOB

folder= '/media/ak/My Passport/Experiment Data/ActivityClockData/'
targetFolder = "".join((folder,'MergedDF/'))
folderList = os.listdir(folder)

symbols =['FB1','JB1','FV1','G_1']
folderList = os.listdir(folder)
def selectFiles(folderList, symbolIDX):
    return list(np.sort([s for s in folderList if ('Clocks') in s and ('_'+str(symbols[symbolIDX])) in s]))

if __name__ == '__main__':
    from collections import defaultdict

    mergedDFs = defaultdict(dict)
    symbolIDX=1
    clocks = selectFiles(folderList, 1)
    clocks= selectFiles(folderList=folderList, symbolIDX=symbolIDX)
    column = 'MicroPrice'
    for clocksIdx, clocks in enumerate(clocks):

        fileLoc = "".join((folder, clocks[clocksIdx]))
        dFclockdf = pkl.load(open(fileLoc, "rb"))
        # sub1= createLOB.get_ohlc(dFclockdf[list(dFclockdf.keys())[0]], dFclockdf[list(dFclockdf.keys())[1]]) #TickBar
        # sub2 = createLOB.get_ohlc(dFclockdf[list(dFclockdf.keys())[0]], dFclockdf[list(dFclockdf.keys())[2]]) #Volume Bar
        # # merging various DataFrames

        mergedDFs['OriginalDF'][clocksIdx ]=dFclockdf[list(dFclockdf.keys())[0]] #append using the original dataframe
        # mergedDFs['rogersSatchellTickBar'][clocksIdx ] = createLOB.Volestim(sub1, window=50).rogersSatchell()
        # mergedDFs['hodgesTompkinsTickBar'][clocksIdx] = createLOB.Volestim(sub1, window=50).hodgesTompkins()
        # mergedDFs['yangZhangTickBar'][clocksIdx ] = createLOB.Volestim(sub1, window=50).yangZhang()
        mergedDFs['TickBarDF'][clocksIdx ] = dFclockdf[list(dFclockdf.keys())[1]] #append using the tick bar dataframe
        mergedDFs['VolumeBarDF'][clocksIdx ] = dFclockdf[list(dFclockdf.keys())[2]] #append using the volume bar dataframe
        mergedDFs['DollarVolumeBarDF'][clocksIdx] = dFclockdf[list(dFclockdf.keys())[3]]  # append using the volume bar dataframe

        # mergedDFs['rogersSatchellVolmBar'][clocksIdx ] = createLOB.Volestim(sub2, window=50).rogersSatchell()
        # mergedDFs['hodgesTompkinsVolmBar'][clocksIdx ] = createLOB.Volestim(sub2, window=50).hodgesTompkins()
        # mergedDFs['yangZhangVolmBar'][clocksIdx ] = createLOB.Volestim(sub2, window=50).yangZhang()
        FileName = "".join(('MergedDataFrame_', str(symbols[symbolIDX])))

        save_loc = os.path.join(targetFolder, FileName)
        pkl.dump(mergedDFs, open(save_loc, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)




