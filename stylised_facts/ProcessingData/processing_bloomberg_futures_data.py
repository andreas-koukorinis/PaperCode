import lob_for_futures as lob
import os
from collections import defaultdict
import pickle

if __name__ == '__main__':
    dataFolder = lob.dataFolder

    symbols = ['VIX', 'FB1', 'TU1', 'G_1', 'RX1', 'OE1', 'TY1', 'FV1',
               'JB1', 'RX1', 'DU1', 'KE1', 'US1', 'YM1', 'XM1', 'VXX']
    trades_cols = ['size', 'time', 'type', 'value']
    quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
    trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]

    symbol = 'US1'
    print(symbol)
    quotes_string = "".join((symbol, "_Comdty_quotes"))
    trades_string = "".join((symbol, "_Comdty_trades"))
    quotesFileCh = os.path.join(dataFolder, quotes_string)
    tradesFileCh = os.path.join(dataFolder, trades_string)

    quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])
    tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])

    print(quotesDates)
    width = 400
    time_resample_rate = "600S"

    testClass = lob.DataLoader(data_location=dataFolder,
                           symbol=str(symbol),
                           dates=quotesDates,
                           use_columns=trades_cols,
                           calendar_resample=str(time_resample_rate),
                           trade_volume_width=width,
                           ticks_width=width,
                           usd_volume_width=width)

    bars_dict = defaultdict(dict)
    for date in quotesDates:
        df = testClass.load_and_format_data()[str(date)]
        input_dict = testClass.get_bars(df)
        calendar_bar_df = (testClass.get_concat_data(testClass._bars_dict)['calendar_bars'])
        tick_bar_df = testClass.get_concat_data(testClass._bars_dict)['tick_bars']
        volume_bar_df = (testClass.get_concat_data(testClass._bars_dict)['volume_bars'])
        USD_volume_bar_df = (testClass.get_concat_data(testClass._bars_dict)['usd_volume_bars'])
        bars_dict['volume_bar'][date] = volume_bar_df
        bars_dict['calendar_bar'][date] = calendar_bar_df
        bars_dict['usd_volume_bar'][date] = USD_volume_bar_df
        bars_dict['tick_bar'][date] = tick_bar_df
        # storage location
        dictFileName = "".join((symbol + '_' + str(date) + '_consolidated_lob_clock_tick_'+str(width)+'_cal_resample_'+str(time_resample_rate) + '.pkl'))
        save_loc = os.path.join('/media/ak/Elements/OrderBookProcessedData/', symbol, dictFileName)
        pickle.dump(bars_dict, open(save_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)








