import numpy as np
import pandas as pd
import json

from utils.custom_logger import logger

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DataLoader:
    def __init__(self,
                 data_location,
                 dates,
                 book_levels,
                 use_cols,
                 features,
                 calendar_resample,
                 ticks_width,
                 trade_volume_width,
                 usd_volume_width,
                 clock_turns):
        self._data_location = data_location
        self._dates = dates
        self._book_levels = book_levels
        self._use_cols = use_cols
        self._features = features
        self._calendar_resample = calendar_resample
        self._ticks_width = ticks_width
        self._trade_volume_width = trade_volume_width
        self._usd_volume_width = usd_volume_width
        self._clock_turns = clock_turns

        self._bars_dict = {}
        self.calendar_bars = []
        self.volume_bars = []
        self.tick_bars = []
        self.usd_volume_bars = []

        self.train_data = {}
        self.test_data = {}

        self._logger = logger('DataLoader')

    def load_data(self):
        for date in self._dates:
            d = date.date()
            self._logger.info(f"Loading data for: {d}")
            tmp_df = pd.read_csv(f"{self._data_location}/{d}.csv", usecols=self._use_cols)
            # TODO: Run using MDReader to speed things up
            self._logger.info(f"Successfully loaded data for: {d}")
            tmp_df = self.apply_micro_structure_features(tmp_df)[self._features].dropna()
            tmp_df = self.remove_outliers(tmp_df)
            self.get_bars(tmp_df)
            self.turn_clocks(input_dict=self._bars_dict)
        self.train_test_split(input_dict=self._bars_dict, train_percent=70)

    def apply_micro_structure_features(self, df):
        self._logger.info("Applying micro-structure features")
        bid_size_list, ask_size_list = [[f"{ba}_size_{t}" for t in range(self._book_levels)] for ba in ['bid', 'ask']]
        bid_price_list, ask_price_list = [[f"{ba}_price_{t}" for t in range(self._book_levels)] for ba in
                                          ['bid', 'ask']]

        bid_prices = df[bid_price_list].values
        ask_prices = df[ask_price_list].values
        bid_sizes = df[bid_size_list].values
        ask_sizes = df[ask_size_list].values

        df.timestamp = df.timestamp.astype('datetime64')
        # df['milliSeconds'] = df.timestamp.dt.microsecond / 0.001
        df['weighted_average_bid'] = (bid_prices * bid_sizes).sum(axis=1) / bid_sizes.sum(axis=1)
        df['weighted_average_ask'] = (ask_prices * ask_sizes).sum(axis=1) / ask_sizes.sum(axis=1)
        df['activity_spread'] = df.weighted_average_ask.values - df.weighted_average_bid.values
        df['total_size'] = (ask_sizes + bid_sizes).sum(axis=1)
        df['micro_price'] = ((ask_prices * ask_sizes).sum(axis=1) + (bid_prices * bid_sizes).sum(
            axis=1)) / df.total_size.values
        df['price_imbalance'] = ((ask_prices * ask_sizes).sum(axis=1) - (bid_prices * bid_sizes).sum(
            axis=1)) / df.total_size.values
        df['mid_simple'] = 0.5 * (
                ask_prices.sum(axis=1) / df.ask_level_count + bid_prices.sum(axis=1) / df.bid_level_count)
        # TODO: removed filter here as looking for "ask_price" was including "weighted_average_ask_price"
        #  same issue for bid
        df['total_traded_volume'] = df.trades_buy_volume.values + df.trades_sell_volume.values
        df['dollar_traded_volume'] = df.trades_sell_volume.values * df.trades_sell_avg_price.values \
                                     + df.trades_buy_volume.values * df.trades_buy_avg_price.values
        # TODO: changed this from VWAP to avg price
        df['total_trades'] = df.trades_buy_count.values + df.trades_sell_count.values
        df['pct_change_micro_price'] = df.micro_price.pct_change()
        # TODO: do we want to be using pct_change here or log returns?
        return df

    def get_bars(self, df):
        self._logger.info('Calculating calendar clock')
        calendar_bars = df.resample(rule=self._calendar_resample, on='timestamp').agg(['first', 'max', 'min', 'last'])
        calendar_bars = self.prep_bars(calendar_bars, 'timestamp')

        self._logger.info('Calculating volume clock')
        # TODO: Doing this on day-by-day basis. We lose the last turn of the clock each day
        df['cum_total_trade_volume'] = df.total_traded_volume.cumsum()
        df['volume_bucket'] = np.floor(df.cum_total_trade_volume / self._trade_volume_width)
        volume_bars = df.groupby(by='volume_bucket').agg(['first', 'max', 'min', 'last']).iloc[1:-1]
        volume_bars = self.prep_bars(volume_bars, 'cum_total_trade_volume')
        df.drop(['cum_total_trade_volume', 'volume_bucket'], axis=1, inplace=True)

        self._logger.info('Calculating tick clock')
        # TODO: ticks here means any change to the order book. Not necessarily a trade. Can add that clock too
        df['tick_count'] = range(len(df))
        df['tick_bucket'] = np.floor(df.tick_count / self._ticks_width)
        tick_bars = df.groupby(by='tick_bucket').agg(['first', 'max', 'min', 'last']).iloc[1:-1]
        tick_bars = self.prep_bars(tick_bars, 'tick_count')
        df.drop(['tick_count', 'tick_bucket'], axis=1, inplace=True)

        self._logger.info('Calculating usd volume clock')
        df['cum_usd_volume'] = df.dollar_traded_volume.cumsum()
        df['usd_volume_bucket'] = np.floor(df.cum_usd_volume / self._usd_volume_width)
        usd_volume_bars = df.groupby(by='usd_volume_bucket').agg(['first', 'max', 'min', 'last']).iloc[1:-1]
        usd_volume_bars = self.prep_bars(usd_volume_bars, 'cum_usd_volume')
        df.drop(['cum_usd_volume', 'usd_volume_bucket'], axis=1, inplace=True)

        self.calendar_bars.append(calendar_bars)
        self.volume_bars.append(volume_bars)
        self.tick_bars.append(tick_bars)
        self.usd_volume_bars.append(usd_volume_bars)

        self._bars_dict['calendar_bars'] = self.calendar_bars
        self._bars_dict['volume_bars'] = self.volume_bars
        self._bars_dict['tick_bars'] = self.tick_bars
        self._bars_dict['usd_volume_bars'] = self.usd_volume_bars

    @staticmethod
    def remove_outliers(df):
        # TODO: Need to implement

        # TODO: Removing outliers- I dont think this should be applied to the whole df. Maybe just some columns?
        #  df = df.clip(upper=df.median() + df.mad(), lower=df.median() - df.mad(), axis=1)
        #  Should we be getting z-scores first here?
        #  is median +- mad and clipping the correct implementation? can do gaussian or mean smoothing too.
        #  Do we do any de-noising?
        #  I think we should rather use a rolling mean + 3*std to get rid of outliers. Just need to determine how long
        #  the rolling period is. Can also use RMS method to calc rolling period
        return df

    @staticmethod
    def prep_bars(df, drop_col):
        df.columns = [f"{i[0]}_{j}" for i, j in zip(df.columns, ['open', 'high', 'low', 'close'] * len(df.columns))]
        if drop_col:
            df.drop([f"{drop_col}_{i}" for i in ['open', 'high', 'low', 'close']], axis=1, inplace=True)
        return df

    def turn_clocks(self, input_dict):
        for df_name in input_dict:
            self._logger.info(f"Turning the clock for {df_name}")
            for t in range(1, self._clock_turns+1):
                input_dict[df_name][-1][f"{t}_clock_turns"] = np.log(
                    input_dict[df_name][-1].shift(-t)['mid_close'] / input_dict[df_name][-1].mid_close)

    def train_test_split(self, input_dict, train_percent):
        dates_args = [i for i in range(len(self._dates))]
        train_days = np.random.choice(a=dates_args, size=int(len(dates_args) * train_percent/100), replace=False)
        test_days = [x for x in dates_args if x not in train_days]

        self._logger.info(f"Using the following for training: {self._dates[train_days]}")
        self._logger.info(f"Using the following for testing: {self._dates[test_days]}")

        for bar in input_dict:
            self.train_data[bar] = pd.concat([input_dict[bar][i] for i in train_days], ignore_index=True)
            self.test_data[bar] = pd.concat([input_dict[bar][i] for i in test_days], ignore_index=True)
        # TODO: dropna here?


if __name__ == '__main__':
    with open('inputs.json') as f:
        input_config = json.load(f)

    data = DataLoader(
        data_location=input_config['location'],
        dates=pd.date_range(start=input_config['dates']['start_date'], periods=input_config['dates']['num_days']),
        book_levels=input_config['levels'],
        use_cols=input_config['cols_to_use'],
        features=input_config['wanted_features'],
        calendar_resample=input_config['widths']['calendar_resample'],
        ticks_width=input_config['widths']['clock_ticks_width'],
        trade_volume_width=input_config['widths']['clock_trade_volume_width'],
        usd_volume_width=input_config['widths']['clock_usd_volume_width'],
        clock_turns=input_config['num_clock_turns']
    )

    data.load_data()
