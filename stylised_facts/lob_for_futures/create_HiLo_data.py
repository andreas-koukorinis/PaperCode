# from multiprocessing import Pool, freeze_support
# import time
# import itertools
from lob_for_futures import *
from pathlib import Path

bars = ['calendar', 'tick', 'dollar', 'volume']
if __name__ == '__main__':
    output_data_folder = os.path.join(augExpertiments, 'HiLoData')

    for symbolIdx in range(0, len(symbols)):

        barCh = 'volume'
        symbolFolder = os.path.join(expInputFiles, symbols[symbolIdx])
        hi_suffix = 'Hi_Vol_Regime'
        lo_suffix = 'Lo_Vol_Regime'

        hi_write_file_dir = os.path.join(output_data_folder, f"{symbols[symbolIdx]}_{hi_suffix}")
        Path(hi_write_file_dir).mkdir(parents=True, exist_ok=True)
        lo_write_file_dir = os.path.join(output_data_folder, f"{symbols[symbolIdx]}_{lo_suffix}")
        Path(lo_write_file_dir).mkdir(parents=True, exist_ok=True)

        barFiles = [f for f in os.listdir(symbolFolder) if str(barCh) in f]
        inaccurate_dates = list()


        def produce_hi_lo_data(idx):
            idxFileLoc = os.path.join(symbolFolder, barFiles[idx])
            date_idx = idxFileLoc.split(barCh)[1].split("_")[1]
            print(idxFileLoc)
            df = pd.read_pickle(idxFileLoc)
            try:
                dfHi, dfLo = [x for _, x in df.groupby(df['GK_vol'] < np.median(df.GK_vol))]
                hi_vol_file_loc = os.path.join(hi_write_file_dir,
                                               str(symbols[symbolIdx]) + "_" + str(barCh) + "_hi_regime" + str(
                                                   date_idx) + ".pkl")
                lo_vol_file_loc = os.path.join(lo_write_file_dir,
                                               str(symbols[symbolIdx]) + "_" + str(barCh) + "_lo_regime" + str(
                                                   date_idx) + ".pkl")

                pickle.dump(dfHi, open(hi_vol_file_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(dfLo, open(lo_vol_file_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                print('saving:', hi_vol_file_loc)

                # print(dfHi.head(10))
                # print(dfLo.head(10))
            except:
                pass
                inaccurate_dates.append(barFiles[idx])  # logs of problematic dates
            # this is the logging bit
            inaccurates_dates_loc = os.path.join(output_data_folder,
                                                 str(symbols[symbolIdx]) + "_" + str(barCh) + "_no_data_list" + ".pkl")
            pickle.dump(inaccurate_dates, open(inaccurates_dates_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        if len(barFiles)!=0:
            print ('ok')
            # load example file

            for idx in range(0, len(barFiles)):
                produce_hi_lo_data(idx)

        else:
            print('not ok')

