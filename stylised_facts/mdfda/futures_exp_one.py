import lob_for_futures as lobFut
from lob_for_futures import *
import os
import matplotlib.pyplot as plt

dataFolder = lobFut.dataFolder


if __name__ == '__main__':
    symbolIdx = 4

    symbol = sorted(symbols)[symbolIdx]
    print(symbol)
    quotesFileCh = os.path.join(dataFolder, quotes[symbolIdx])
    tradesFileCh = os.path.join(dataFolder, trades[symbolIdx])

    # get common Dates
    quotesDates = sorted([f.split(".csv")[0] for f in os.listdir(quotesFileCh)])
    tradesDates = sorted([f.split(".csv")[0] for f in os.listdir(tradesFileCh)])
    intersectionDates = list(set(quotesDates).intersection(tradesDates))

    trades_cols = ['size', 'time', 'type', 'value']

    # params for the clocks

    calendar_resample_freq = "900S"
    trade_volume_width = 400
    ticks_width = 400
    usd_volume_width = 400
    dates_choice = intersectionDates[0:20]

    testClass = DataLoader(data_location=dataFolder,
                           symbol=symbol,
                           dates=dates_choice,
                           use_columns=trades_cols,
                           calendar_resample=calendar_resample_freq,
                           trade_volume_width=trade_volume_width,
                           ticks_width=ticks_width,
                           usd_volume_width=usd_volume_width)

    hash_of_file = "_".join(
        (str(symbol), "volume_width", str(trade_volume_width), "calendar_resample", str(calendar_resample_freq)))

    # load data
    input_dict = testClass.load_and_format_data()

    tick_bar_dfs = []
    volume_bar_dfs = []
    usd_volume_bar_dfs = []
    calendar_bar_dfs = []
    dates = list(input_dict.keys())
    for date in dates:
        df = testClass.load_and_format_data()[str(date)]
        input_dict = testClass.get_bars(df)
        tick_bar_df = testClass.get_concat_data(testClass._bars_dict)['tick_bars']
        volume_bar_df = testClass.get_concat_data(testClass._bars_dict)['volume_bars']
        usd_volume_bar_df = testClass.get_concat_data(testClass._bars_dict)['usd_volume_bars']
        calendar_bar_df = testClass.get_concat_data(testClass._bars_dict)['calendar_bars']
        tick_bar_dfs.append(tick_bar_df)
        volume_bar_dfs.append(volume_bar_df)
        usd_volume_bar_dfs.append(usd_volume_bar_df)
        calendar_bar_dfs.append(calendar_bar_df)

    consolidated_tick_bar_df = pd.concat(tick_bar_dfs).dropna()
    consolidated_volume_bar_df = pd.concat(volume_bar_dfs).dropna()
    consolidated_calendar_bar_df = pd.concat(calendar_bar_dfs).dropna()
    consolidated_usd_volume_bar_df = pd.concat(usd_volume_bar_dfs).dropna()

    bars = {'Volume bars clock': consolidated_volume_bar_df,
            'Tick bars clock': consolidated_tick_bar_df, 'USDVolume bars clock':
                consolidated_usd_volume_bar_df,
            'Chrono clock': consolidated_calendar_bar_df}
    barsKeys = list(bars.keys())

    vr = returns(bars[barsKeys[1]].micro_price_close).replace([np.inf, -np.inf], 0)  # volume
    tr = returns(bars[barsKeys[0]].micro_price_close).replace([np.inf, -np.inf], 0)  # tick
    dr = returns(bars[barsKeys[2]].micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # usd volume
    df_ret = returns(bars[barsKeys[3]].micro_price_close).dropna().replace([np.inf, -np.inf], 0)  # calendar
    bar_returns = [tr, vr, dr, df_ret]
    bar_types = ['tick', 'volume', 'dollar', 'calendar']

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.kdeplot(tr,
                ax=ax,
                label="Tick",
                bw=0.25,
                color='darkblue')
    sns.kdeplot(dr,
                ax=ax,
                label="USD Volume",
                bw=0.55, linewidth=2.25,
                color='blue', linestyle=':')

    sns.kdeplot(vr,
                ax=ax,
                label="Volume",
                bw=0.55, linewidth=1.25,
                color='red', linestyle='--')

    plt.xlabel('Returns', fontsize=9)
    plt.ylabel('Density', fontsize=9)
    plt.xticks(fontsize=9, rotation=45)
    plt.yticks(fontsize=9)
    plt.title(str(symbol))
    file_name = str(hash_of_file) + '_multiclocks_density_plot.png'
    plt.savefig(os.path.join(figures_destination, file_name))
    plt.legend()
    plt.show()



