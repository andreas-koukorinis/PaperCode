import os
import pandas as pd
from pathlib import Path
import pickle


class DataAnalyzer:
    def __init__(self, information_clock_dir, symbol, bar_choice, mfdfa_vars):
        """
        Initialize the DataAnalyzer class with the required parameters.

        Args:
        - information_clock_dir (str): The directory containing information clock files.
        - symbol (str): The symbol to analyze.
        - bar_choice (str): The bar choice to filter files by.
        - mfdfa_vars (list): A list of MFDFA variables to filter files by.
        """
        self.symbol = symbol
        self.information_clock_dir = information_clock_dir
        self.bar_choice = bar_choice
        self.mfdfa_vars = mfdfa_vars
        self.symbol_inform_clock_dir = os.path.join(information_clock_dir, symbol)

    def get_symbols(self):
        """
        Get a list of symbols from the information clock directory.

        Returns:
        - list: A list of sorted symbols.
        """
        return sorted(os.listdir(self.information_clock_dir))

    def get_symbol_files(self):
        """
        Get a list of files for the current symbol.

        Returns:
        - list: A list of file names.
        """
        return sorted(os.listdir(self.symbol_inform_clock_dir))

    def get_files_by_keyword(self, files, keyword):
        """
        Get a list of files that contain a specific keyword.

        Args:
        - files (list): A list of file names.
        - keyword (str): The keyword to filter files by.

        Returns:
        - list: A list of filtered file names.
        """
        return [f for f in files if keyword in f]

    def analyze(self):
        """
        Analyze data for the given symbol.

        Returns:
        - dict: A dictionary with the analysis result.
        """
        files_filtered_by_bar_choice = self.get_files_by_keyword(self.get_symbol_files(), self.bar_choice)

        if not files_filtered_by_bar_choice:
            raise ValueError(f"No files found with bar_choice='{self.bar_choice}'.")

        analysis_results = {}

        for mfdfa_var in self.mfdfa_vars:
            mfdfa_var_dict = {}

            for idx, file in enumerate(files_filtered_by_bar_choice):
                file_content = pd.read_pickle(os.path.join(self.symbol_inform_clock_dir, file))
                mfdfa_var_value = file_content[str(self.bar_choice)].get(str(mfdfa_var))

                if mfdfa_var_value is not None:
                    mfdfa_var_dict[idx] = mfdfa_var_value

            analysis_results[mfdfa_var] = mfdfa_var_dict

        return analysis_results

    def save_analysis_results(self, analysis_results, target_dir, file_name):
        """
        Save the analysis results to a specified target directory with a given file name.

        Args:
        - analysis_results (dict): A dictionary with the analysis result.
        - target_dir (str): The directory to save the analysis results.
        - file_name (str): The file name to save the analysis results.
        """
        target_path = os.path.join(target_dir, file_name)
        with open(target_path, 'wb') as f:
            pickle.dump(analysis_results, f)
        print(f"Analysis results saved to: {target_path}")


if __name__ == '__main__':
    # Define the required parameters
    information_clock_dir = '/media/ak/T7/August11th2022Experiments/InfoClockDataFrames'
    symbol = 'XM1'
    bar_choice = 'dollar'
    variables = ['n_F', 'list_H', 'list_H_intercept', 'tau', 'alpha', 'mfSpect'] # MFDFA variables you're interested in
    # Create an instance of the DataAnalyzer class
    analyzer = DataAnalyzer(information_clock_dir, symbol, bar_choice, variables)

    # Analyze the data for the given symbol
    analysis_result = analyzer.analyze()

    # save results
    LinearMMDInputFiles = '/media/ak/T7/August11th2022Experiments/LinearMMDInputFiles'
    file_name = "_".join((symbol, bar_choice, 'mfdfa_dicts_all_dates.pkl'))
    print(file_name)
    analyzer.save_analysis_results(analysis_result, LinearMMDInputFiles, file_name)
