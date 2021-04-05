import pandas as pd
from custom_logger import logger


class ClockWidth:
    def __init__(self, data_location, date_range, seconds_split):
        self._data_location = data_location
        self._date_range = date_range
        self._seconds_split = seconds_split

        self.usd_volume_width = None
        self.ticks_width = None
        self.trade_volume_width = None

        self._logger = logger('ClockWidth')

    def get_widths(self):
        num_ticks = 0
        trade_volume = 0
        usd_volume = 0
        for date in self._date_range:
            d = date.date()
            self._logger.info(f"Pulling data for: {d}")
            df = pd.read_csv(f"{self._data_location}/{d}.csv.gz", usecols=['trades_buy_volume',
                                                                           'trades_sell_volume',
                                                                           'trades_buy_avg_price',
                                                                           'trades_sell_avg_price'])
            num_ticks += len(df)
            trade_volume += (df.trades_buy_volume.values + df.trades_sell_volume.values).sum()
            usd_volume += (
                    df.trades_sell_volume.values * df.trades_sell_avg_price.values +
                    df.trades_buy_volume.values * df.trades_buy_avg_price.values).sum()
        self._logger.info(f"Total usd volume: {usd_volume:,.2f}")
        self._logger.info(f"Total number of ticks: {num_ticks:,.2f}")
        self._logger.info(f"Total traded volume: {trade_volume:,.2f}")

        time_splits = 86400 * len(self._date_range) / self._seconds_split

        self._logger.info(f"There are {len(self._date_range)} days in the sample")
        self._logger.info(f"Using a {self._seconds_split} minute calendar interval, "
                          f"we split the data into {time_splits} parts")

        self.usd_volume_width = usd_volume / time_splits
        self.ticks_width = num_ticks / time_splits
        self.trade_volume_width = trade_volume / time_splits

        self._logger.info(f"Recommended usd_volume_width: {self.usd_volume_width:,.2f}:")
        self._logger.info(f"Recommended ticks_width: {self.ticks_width:,.2f}")
        self._logger.info(f"Recommended trade_volume width: {self.trade_volume_width:,.2f}")


if __name__ == '__main__':
    clock_widths = ClockWidth(
        data_location='/Users/joshuajacobsohn/Desktop/DBT_data/',
        date_range=pd.date_range(start='2020-10-01', periods=3),
        seconds_split=300
    )
    clock_widths.get_widths()
