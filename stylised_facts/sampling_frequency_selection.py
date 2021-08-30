import lob_for_futures as lob
from lob_for_futures import dataFolder, DataLoader, returns, defaultdict, pickle
import os
import numpy as np

from scipy.stats import shapiro, normaltest, jarque_bera


def normality_tests(data_values):
    """

    :param data_values: values of returns in our case
    :return: print out a series of outcomes of whether the data fits a normal distribution or not!
    """

    stat, p = shapiro(data_values)
    print('stat = %.3f, p = %.3f\n ' % (stat, p))
    if p > 0.05:
        print('prob gaussian')
    else:
        print('non gaussian')

    stat_nt, p_nt = normaltest(data_values)

    print('stat = %.3f, p = %.3f\n ' % (stat_nt, p_nt))

    stat_jb, p_jb = jarque_bera(data_values)

    print('stat = %.3f, p = %.3f\n ' % (stat_jb, p_jb))

    if p_jb> 0.05:
        print('prob gaussian')
    else:
        print('non gaussian')


if __name__ == '__main__':
    symbols = ['VIX', 'FB1', 'TU1', 'G_1', 'RX1', 'OE1', 'TY1', 'FV1',
               'JB1', 'RX1', 'DU1', 'KE1', 'US1', 'YM1', 'XM1', 'VXX']
    trades_cols = ['size', 'time', 'type', 'value']
    quotes = [f for f in os.listdir(dataFolder) if str('_quotes') in f]
    trades = [f for f in os.listdir(dataFolder) if str('_trades') in f]

    symbol = 'OE1'
    print(symbol)
    quotes_string = "".join((symbol, "_Comdty_quotes"))
    trades_string = "".join((symbol, "_Comdty_trades"))
    quotesFileCh = os.path.join(dataFolder, quotes_string)
    tradesFileCh = os.path.join(dataFolder, trades_string)

    quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])

    tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])

    print(quotesDates)

    potential_widths =[40, 50, 60, 75, 90, 100, 125, 150]
    sample_dates = quotesDates
    tick_sampling_dict = defaultdict(dict)
    volume_sampling_dict = defaultdict(dict)
    for width in potential_widths:
        testClass = DataLoader(data_location=dataFolder,
                               symbol=str(symbol),
                               dates=sample_dates,
                               use_columns=trades_cols,
                               calendar_resample="600S",
                               trade_volume_width=width,
                               ticks_width=width,
                               usd_volume_width=width)
        for date in sample_dates:
            df = testClass.load_and_format_data()[str(date)]
            input_dict = testClass.get_bars(df)
            tick_bar_df = testClass.get_concat_data(testClass._bars_dict)['tick_bars']
            volume_bar_df = testClass.get_concat_data(testClass._bars_dict)['volume_bars']

            tr = np.isfinite(returns(tick_bar_df.micro_price_close).dropna())
            vr = np.isfinite(returns(tick_bar_df.micro_price_close).dropna())
            tick_standard = (tr - tr.mean()) / tr.std()
            vr_standard = (vr - vr.mean()) / vr.std()

            print("doing tick clock data for width", width, " and date ", date)
            normality_tests(tr)
            tick_sampling_dict[str(date)][width] = [ (normaltest(tr)), (jarque_bera(tr)), (shapiro(tr)) ]
            print("doing volume clock data for width", width, " and date ", date)
            volume_sampling_dict[str(date)][width] = [(normaltest(vr)), (jarque_bera(vr)), (shapiro(vr))]

        dictFileName = "".join(
                        (symbol + '_' + '_resampling_tick' + '.pkl'))
        save_loc = os.path.join('/media/ak/Elements/OrderBookProcessedData/', symbol, dictFileName)
        pickle.dump(tick_sampling_dict, open(save_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        dictFileName = "".join(
                        (symbol + '_' + '_resampling_volume' + '.pkl'))
        save_loc = os.path.join('/media/ak/Elements/OrderBookProcessedData/', symbol, dictFileName)
        pickle.dump(volume_sampling_dict, open(save_loc, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



