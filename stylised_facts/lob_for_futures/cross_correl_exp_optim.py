import concurrent.futures
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List
from typing import Tuple

import fathon
import numpy as np
import pandas as pd
from fathon import fathonUtils as fu

SYMBOLS = ['RX1', 'FB1', 'JB1', 'G_1', 'FV1', 'TY1', 'TU1', 'DU1', 'YM1', 'XM1', 'US1', 'OE1', 'KE1']
WIN_SIZES = fu.linRangeByStep(10, 1000, step=20)
POL_ORD = 1
MICRO_VARIABLES = ['arrival_rates', 'gk_vol', 'median_traded_volume', 'micro_price_change']


class MicroVariableProcessor:
    def __init__(self, base_path: str, symbol: str, bar: str):
        """
        Initialize the MicroVariableProcessor class.

        :param base_path: The base path to the directory containing the experiment data.
        :type base_path: str
        :param symbol: The symbol of the financial instrument (e.g., 'JB1').
        :type symbol: str
        :param bar_mfdfa_files: A list of file names containing the bar-mfdfa data.
        :type bar_mfdfa_files: List[str]
        """
        self.base_path = base_path
        self.symbol = symbol
        self.bar = bar
        self.symbol_path = os.path.join(self.base_path, self.symbol)
        self.bar_mfdfa_files = [f for f in os.listdir(self.symbol_path) if str(self.bar) in f]

        self.gk_vol_dict = defaultdict(dict)
        self.micro_prices = None
        self.median_traded_volume_dict = defaultdict(dict)
        self.arrival_rates_dict = defaultdict(dict)
        self.micro_price = defaultdict(dict)

    def compute_intraday_volatility(self, micro_prices: pd.Series) -> float:
        """
        Compute the intraday volatility of returns based on micro prices.

        :param micro_prices: A Pandas Series containing micro price data.
        :type micro_prices: pd.Series
        :return: Intraday volatility of returns.
        :rtype: float
        """
        log_returns = np.log(micro_prices / micro_prices.shift(1))
        intraday_volatility = np.std(log_returns)

        return intraday_volatility

    def process_file(self, idx: int) -> Dict:
        micro_variables = ['arrival_rates', 'GK_vol', 'median_traded_volume', 'micro_price_change']
        file_path = os.path.join(self.symbol_path, self.bar_mfdfa_files[idx])
        pkl_dict = pd.read_pickle(file_path)
        #         pkl_dict_keys = sorted(list(pkl_dict.keys()))

        output = {
            "gk_vol": pkl_dict['GK_vol'],
            "median_traded_volume": pkl_dict["median_traded_volume"],
            "arrival_rates": pkl_dict["arrival_rates"],
            "micro_price": pkl_dict["micro_price"],

        }

        return output

    def compute_std_and_quantiles(self, intraday_volatilities: List[float]):
        """
        Compute standard deviations and quantiles for the input list of intraday volatilities.

        Args:
            intraday_volatilities (List[float]): A list of intraday volatilities.

        Returns:
            tuple: A tuple containing the standard deviation DataFrame, first regime quantile, and second regime quantile.
        """
        mpc_df = pd.DataFrame(intraday_volatilities, columns=["intraday_volatility"])
        std_df = pd.DataFrame()
        std = mpc_df.std(axis=0)
        std_df['std'] = std
        first_regime = np.quantile(std, 0.33)
        second_regime = np.quantile(std, 0.66)

        return std_df, first_regime, second_regime

    def process_files(self):
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(self.process_file, idx): idx for idx, file in
                             enumerate(self.bar_mfdfa_files)}

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    file_data = future.result()
                except Exception as exc:
                    print(f"File {idx} generated an exception: {exc}")
                else:
                    results.append((idx, file_data))

        for idx, file_data in results:
            try:
                self.gk_vol_dict[idx] = file_data["gk_vol"]
                self.median_traded_volume_dict[idx] = file_data["median_traded_volume"]
                self.arrival_rates_dict[idx] = file_data["arrival_rates"]
                self.micro_price[idx] = file_data["micro_price"]
            except KeyError:
                pass

        return self.gk_vol_dict, self.median_traded_volume_dict, self.arrival_rates_dict, self.micro_price

    def get_output(self, idx):
        """
        Return the computed intraday volatility and other attributes.

        :param idx: The index of the file to be processed.
        :type idx: int
        :return: A dictionary containing the computed intraday volatility and other attributes.
        :rtype: dict
        """
        # Process the specified file
        file_data = self.process_file(idx)

        # Extract the micro_price data
        micro_prices = file_data["micro_price"]

        # Set the micro_prices attribute
        self.micro_prices = micro_prices

        output = {
            "intraday_volatility": self.compute_intraday_volatility(self.micro_prices),
            "gk_vol": file_data["gk_vol"],
            "median_traded_volume": file_data["median_traded_volume"],
            "arrival_rates": file_data["arrival_rates"],
            "micro_price": file_data["micro_price"]
        }

        return output

    def save_output(self, filepath: str):
        """
        Save the output to a specified file.

        Args:
            filepath (str): The path where the output should be saved.
        """
        output = self.get_output()

        with open(filepath, "w") as f:
            json.dump(output, f, indent=4)


class CrossCorrel:
    def __init__(self, base_path: str, symbols_list: list, idx: int, bar_choice: str):
        """
        Initialize the CrossCorrel class.

        Args:
            base_path (str): Path to read data from.
            symbols_list (list): List of symbols.
            idx (int): Index for the symbol.
            bar_choice (str): Bar for information clock.
        """
        self.base_path = base_path
        self._idx = idx
        self._symbols = symbols_list
        self._bar = bar_choice
        self._symbol = self._symbols[self._idx]
        self.symbol_path = os.path.join(self.base_path, "ExperimentOne", self._symbol[0])
        self.winSizes = fu.linRangeByStep(10, 1000, step=20)
        self.polOrd = 1

    @staticmethod
    def to_aggregated(data):
        """
        Convert pandas Series to numpy array

        Args:
            data (pd.Series): The data to be converted

        Returns:
            np.ndarray: The converted array
        """
        data = np.array(data)
        return data.cumsum()

    # @staticmethod
    def compute_n_rho(var_a: pd.Series, var_b: pd.Series, winSizes: np.ndarray, polOrd: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Compute the DCCA cross-correlation coefficients (rho) and the corresponding window sizes (n).

        Args:
            var_a (pd.Series): The first time series.
            var_b (pd.Series): The second time series.
            win_sizes (np.ndarray): An array of window sizes to be used for DCCA.
            pol_ord (int): The polynomial order for detrending.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The window sizes (n) and the cross-correlation coefficients (rho).
        """
        a = CrossCorrel.to_aggregated(var_a.values)
        b = CrossCorrel.to_aggregated(var_b.values)

        pydcca = fathon.DCCA(a, b)
        try:
            n, _ = pydcca.computeFlucVec(winSizes=winSizes, polOrd=polOrd)
            _, rho = pydcca.computeRho(winSizes=winSizes, polOrd=polOrd)
        except ZeroDivisionError:
            pass

        return n, rho

    @staticmethod
    def compute_n_f_dcca(var_a: pd.Series, var_b: pd.Series, winSizes: np.ndarray, polOrd: int) -> Tuple[
        np.ndarray, np.ndarray]:
        a = CrossCorrel.to_aggregated(var_a)
        b = CrossCorrel.to_aggregated(var_b)

        pydcca = fathon.DCCA(a, b)
        n, F = pydcca.computeFlucVec(winSizes=winSizes, polOrd=polOrd)

        return n, F

    @staticmethod
    def compute_h_h_intc_dcca(var_a: pd.Series, var_b: pd.Series, winSizes: np.ndarray, polOrd: int) -> Tuple[
        np.ndarray, np.ndarray, float, float]:
        a = CrossCorrel.to_aggregated(var_a)
        b = CrossCorrel.to_aggregated(var_b)

        pydcca = fathon.DCCA(a, b)
        n, F = pydcca.computeFlucVec(winSizes=winSizes, polOrd=polOrd)

        H, H_intercept = pydcca.fitFlucVec()

        return n, F, H, H_intercept

    @staticmethod
    def save_output(output_to_save, filepath: str):
        """
        Save the output to a specified file.

        Args:
            filepath (str): The path where the output should be saved.
        """

        with open(filepath, "w") as f:
            json.dump(output_to_save, f, indent=4)


def main():
    base_path = "/media/ak/Data1/ExperimentData/August11th2022Experiments/ExperimentInputFiles"
    # Define the path     to     the     data and the     list     of     symbols.
    symbols_list = ["G_1"]
    bar_choice = "tick"  # Replace with your choice of bar (e.g., 'tick', 'volume', 'dollar').
    # # Initialize the MicroVariableProcessor.
    mvp = MicroVariableProcessor(base_path, symbols_list[0], bar_choice)
    idx = 0
    output = mvp.get_output(idx)
    print(output)

    # Optional: Save output to a file
    # filepath = "/path/to/save/output.json"
    # mvp.save_output(filepath)


def main_cross_correl(base_path: str, symbols_list: List[str], idx: int, bar_choice: str, var_a: pd.Series,
                      var_b: pd.Series, fileIdx: int):
    """
    Main function to run CrossCorrel for a single symbol.

    Args:
        base_path (str): Path to read data from.
        symbols_list (List[str]): List of symbols.
        idx (int): Index for the symbol.
        bar_choice (str): Bar for information clock.
        var_a (pd.Series): The first time series.
        var_b (pd.Series): The second time series.
        fileIdx (int): The index of the file being processed.
    """
    cross_correl = CrossCorrel(base_path, symbols_list, idx, bar_choice)
    win_sizes = cross_correl.winSizes
    pol_ord = cross_correl.polOrd
    try:

        n, rho = CrossCorrel.compute_n_rho(var_a=var_a, var_b=var_b, winSizes=win_sizes, polOrd=pol_ord)
        n, F = CrossCorrel.compute_n_f_dcca(var_a=var_a, var_b=var_b, winSizes=win_sizes, polOrd=pol_ord)
        n, F, H, H_intercept = CrossCorrel.compute_h_h_intc_dcca(var_a=var_a, var_b=var_b, winSizes=win_sizes,
                                                                 polOrd=pol_ord)

        # Process the results as required, e.g., save to file or display results
        print("n:", n)
        print("rho:", rho)
        print("F:", F)
        print("H:", H)
        print("H_intercept:", H_intercept)
        output_path = os.path.join(base_path, "outputs")
        os.makedirs(output_path, exist_ok=True)
        output_filename = f"{symbols_list[idx]}_{bar_choice}_{fileIdx}_output.json"
        output_filepath = os.path.join(output_path, output_filename)

        output_data = {
            "n": n.tolist(),
            "rho": rho.tolist(),
            "F": F.tolist(),
            "H": H,
            "H_intercept": H_intercept
        }

        with open(output_filepath, "w") as f:
            json.dump(output_data, f, indent=4)
            print("saved")
    except (ValueError, UnboundLocalError):
        pass


if __name__ == "__main__":
    import time

    base_path = "/media/ak/Data1/ExperimentData/August11th2022Experiments/ExperimentInputFiles"
    # Define the path     to     the     data and the     list     of     symbols.
    symbols_list = ["G_1"]
    bar_choice = "tick"  # Replace with your choice of bar (e.g., 'tick', 'volume', 'dollar').
    idx = 0
    mvp = MicroVariableProcessor(base_path, symbols_list[idx], bar_choice)

    tic = time.perf_counter()

    with Pool() as pool:
        pool.starmap(main_cross_correl, [(base_path, symbols_list, idx, bar_choice,
                                          mvp.get_output(fileIdx)["micro_price"], mvp.get_output(fileIdx)["gk_vol"],
                                          fileIdx) for
                                         fileIdx in range(0, 134)])
    toc = time.perf_counter()
    print("elapsed time:", (toc - tic))
