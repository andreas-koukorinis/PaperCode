import os
import json
import pandas as pd


def create_parameter_sets(base_input_folder, base_output_folder, symbols, event_clock_destinations, threshold_values,
                          columns_to_resample, exclude_median_columns):
    """
    Creates parameter sets for event clock resampling.

    Args:
        base_input_folder (str): Base input folder containing subfolders for each symbol.
        base_output_folder (str): Base output folder.
        symbols (List[str]): List of symbols.
        event_clock_destinations (Dict[str, str]): Dictionary mapping event clocks to their destinations.
        threshold_values (List[int]): List of threshold values for each event clock.
        columns_to_resample (List[str]): Columns to be resampled.
        exclude_median_columns (List[str]): Columns to exclude from median calculation.

    Returns:
        List[Dict[str, str]]: List of parameter sets.

    """
    parameter_sets = []

    for symbol in symbols:
        input_folder = os.path.join(base_input_folder, symbol)
        output_folder = os.path.join(base_output_folder, symbol)

        # Collect all files in the input folder
        input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.pkl')]

        for event_clock, event_clock_destination, threshold_value in zip(event_clock_destinations.keys(),
                                                                         event_clock_destinations.values(),
                                                                         threshold_values):
            for input_file in input_files:
                params = {
                    'input_file': input_file,
                    'output_folder': os.path.join(output_folder, event_clock_destination),
                    'event_clock_column': event_clock,
                    'event_clock_threshold': threshold_value,
                    'columns_to_resample': columns_to_resample,
                    'exclude_median_columns': exclude_median_columns,
                }
                parameter_sets.append(params)

    return parameter_sets


# Define the parameters
base_input_folder = '/media/ak/Data/InterestRateFuturesData/ReconstructedLOB'
base_output_folder = '/media/ak/Data/InterestRateFuturesData/EventClocksFiles'
symbols = ['RX1','DU1', 'KE1']
event_clock_destinations = {
    'NoOfTrades': 'tick',
    'TradedVolume': 'volume',
    'CCYTradedVolume': 'dollar',
    # Add more event clocks and their destinations if needed
}
threshold_values = [1, 100, 1000]  # Specify threshold values for each event clock
columns_to_resample = ['time', 'BestBid', 'BestAsk', 'MicroPrice', 'arrival_rate', 'MeanRelativeTickVolume',
                       'OrderImbalance']
exclude_median_columns = []

# Create the parameter sets
parameter_sets = create_parameter_sets(base_input_folder, base_output_folder, symbols, event_clock_destinations,
                                       threshold_values, columns_to_resample, exclude_median_columns)
print(parameter_sets)
# Save parameter sets to a JSON file
config_filepath = '/media/ak/Data/InterestRateFuturesData/EventClocksFiles/configmany.json'
with open(config_filepath, 'w') as f:
    json.dump(parameter_sets, f, indent=4)
