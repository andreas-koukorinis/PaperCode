import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.ticker as mtick


def plot_median_with_std(dataframe, output_dir, output_filename, xlabel, ylabel, title):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the median and standard deviation of the DataFrame across columns
    median_values = dataframe.median()
    std_values = dataframe.std()

    # Set Seaborn plot style
    sns.set(style="whitegrid")

    # Increase the figure size for better readability
    plt.figure(figsize=(10, 6))

    # Create the plot using Seaborn lineplot with marker 'o' and dotted line
    sns.lineplot(data=median_values, marker='o', linewidth=2, linestyle='--')

    # Add shaded area representing the standard deviation
    x = np.arange(len(median_values))
    plt.fill_between(x, (median_values - std_values), (median_values + std_values), color='gray', alpha=0.2)

    # Set axes labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Create a legend
    plt.legend(["Median"])

    #     # Save the plot to the specified directory
    #     plt.savefig(os.path.join(output_dir, output_filename), dpi=300)

    # Show the plot
    plt.show()

def plot_median(dataframe, output_dir, output_filename, xlabel, ylabel, title):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the median of the DataFrame across columns
    median_values = dataframe.median()

    # Set Seaborn plot style
    sns.set(style="whitegrid")

    # Increase the figure size for better readability
    plt.figure(figsize=(10, 6))

    # Create the plot using Seaborn lineplot with marker 'o' and dotted line
    sns.lineplot(data=median_values, marker='o', linewidth=2, linestyle='--')

    # Set axes labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Create a legend
    plt.legend(["Median"])

    # Save the plot to the specified directory
#     plt.savefig(os.path.join(output_dir, output_filename), dpi=300)

    # Show the plot
    plt.show()

# Example usage
# dataframe = pd.DataFrame({"Column1": [1, 2, 3, 4, 5],
#                           "Column2": [2, 4, 6, 8, 10],
#                           "Column3": [3, 6, 9, 12, 15]})

# plot_median(dataframe,
#             output_dir="plots",
#             output_filename="example_median_plot.png",
#             xlabel="Columns",
#             ylabel="Median Value",
#             title="Example Median Plot")