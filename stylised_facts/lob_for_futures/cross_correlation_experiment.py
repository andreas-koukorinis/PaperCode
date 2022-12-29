import numpy as np
import matplotlib.pyplot as plt
import fathon
import pandas as pd
from fathon import fathonUtils as fu
from collections import defaultdict
import lob_for_futures as lobFut
import os
import time
dataFolder = lobFut.dataFolder
t7 =lobFut.t7folder
expInputFiles = os.path.join(lobFut.augExpertiments, 'ExperimentInputFiles')
HiLoData = os.path.join(lobFut.augExpertiments, 'HiLoData')
plt.style.use(os.path.join('/home/ak/.config/matplotlib', 'latexstyle3.mplstyle'))
symbols = ['RX1', 'FB1', 'JB1', 'G_1', 'FV1', 'TY1', 'TU1', 'DU1', 'YM1', 'XM1', 'US1', 'OE1', 'KE1']
if __name__ == '__main__':
    symbolIdx = 10
    symbol = symbols[symbolIdx]

    hi_vol_data = os.path.join(HiLoData, str(symbol)+'_Hi_Vol_Regime')

    lo_vol_data = os.path.join(HiLoData, str(symbol) + '_Lo_Vol_Regime')

    barCh = 'tick'

    barCh_vol_dates = [f.split("e")[2].split(".")[0] for f in [f for f in sorted(os.listdir(lo_vol_data)) if str(barCh) in f]]

    if len(barCh_vol_dates)!=0:
        idx = 0
        def compute_dcca(idx):
        dcca_dict = defaultdict(dict)
        start_time = time.time()
        date = barCh_vol_dates[idx]
        print(date)
        hi_regime_filename_loc = os.path.join(hi_vol_data,"_".join((str(symbol),str(barCh),"hi","regime"+date+".pkl")))
        lo_regime_filename_loc = os.path.join(lo_vol_data,"_".join((str(symbol), str(barCh), "lo", "regime" + date + ".pkl")))
        df1 = pd.read_pickle(hi_regime_filename_loc)
        df2 = pd.read_pickle(lo_regime_filename_loc)

        a1 = fu.toAggregated(df1['price_imbalance'])
        # b1 = df1['median_traded_volume']
        c1 = fu.toAggregated(df1['arrival_rates'])

        a2 = fu.toAggregated(df2['price_imbalance'])
        # b2 = df2['median_traded_volume']
        c2 = fu.toAggregated(df2['arrival_rates'])

        pydcca_a1c1 = fathon.DCCA(a1, c1)
        pydcca_a2c2 = fathon.DCCA(a2, c2)

        winSizes = fu.linRangeByStep(20, 1000, step=50)
        polOrd = 1

        n_a1c1, F_a1c1 = pydcca_a1c1.computeFlucVec(winSizes, polOrd=polOrd)
        n_a2c2, F_a2c2 = pydcca_a2c2.computeFlucVec(winSizes, polOrd=polOrd)

        H_a1c1, H_intercept_a1c1 = pydcca_a1c1.fitFlucVec()
        H_a2c2, H_intercept_a2c2 = pydcca_a1c1.fitFlucVec()

        limits_list = np.array([[20, 120], [220, 870]], dtype=int)
        list_H_a1c1, list_H_intercept_a1c1 = pydcca_a1c1.multiFitFlucVec(limits_list)
        list_H_a2c2, list_H_intercept_a2c2 = pydcca_a1c1.multiFitFlucVec(limits_list)

        n_rho_a1c1, rho_a1c1 = pydcca_a1c1.computeRho(winSizes, polOrd=polOrd)
        n_rho_a2c2, rho_a2c2 = pydcca_a2c2.computeRho(winSizes, polOrd=polOrd)

        dcca_dict[idx]['winSizes'] =winSizes
        dcca_dict[idx]['pol'] = polOrd
        dcca_dict[idx]['n_a1c1'] = n_a1c1
        dcca_dict[idx]['n_a2c2'] = n_a2c2
        dcca_dict[idx]['H_a1c1'] = H_a1c1
        dcca_dict[idx]['H_a2c2'] = H_a2c2
        dcca_dict[idx]['H_intercept_a1c1'] = H_intercept_a1c1
        dcca_dict[idx]['H_intercept_a2c2'] = H_intercept_a2c2
        dcca_dict[idx]['list_H_a1c1'] = list_H_a1c1
        dcca_dict[idx]['list_H_a2c2'] = list_H_a2c2
        dcca_dict[idx]['list_H_intercept_a1c1'] = list_H_intercept_a1c1
        dcca_dict[idx]['list_H_intercept_a2c2'] = list_H_intercept_a2c2
        dcca_dict[idx]['n_rho_a1c1'] = n_rho_a1c1
        dcca_dict[idx]['n_rho_a2c2'] = n_rho_a2c2
        dcca_dict[idx]['limits_list'] = limits_list

        # the confidence intervals you can do seperate
        end_time = time.time()
        print(dcca_dict)
        print(end_time - start_time)

    else:
        pass





