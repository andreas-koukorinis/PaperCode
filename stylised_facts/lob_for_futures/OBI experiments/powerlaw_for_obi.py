import os
import pandas as pd
import pickle
import powerlaw
import matplotlib.pyplot as plt
import concurrent.futures

# Constants
reconLOBs = '/media/ak/Data1/InterestRateFuturesData/ReconstructedLOB'
powerLawFiguresLocation = '/home/ak/Documents/Research/Papers/StylisedFactsPaper/figures/PowerLaw'
powerLawResults = '/home/ak/Documents/Research/Papers/StylisedFactsPaper/PowerLawResults'
directory = 'OrderBookImbalance'
path = os.path.join(reconLOBs, directory)
obiFiles = [f for f in os.listdir(path) if 'calendar.pkl' in f]


def extract_obi_avg(dataframes_dict):
    obi_avg_columns = []
    for df_name, df in dataframes_dict.items():
        if 'OBI_avg' in df.columns:
            obi_avg_col = df['OBI_avg'].rename(f'{df_name}_OBI_avg')
            obi_avg_columns.append(obi_avg_col)
    return pd.concat(obi_avg_columns, axis=1)

def process_file(idx):
    """
    Processes the file at a given index. Performs power-law fitting, creates plots, and saves the results.

    Parameters:
    idx (int): Index of the file to be processed.

    Returns:
    tuple: A tuple containing the index and the symbol associated with the processed file.
    """
    # Load data from the file at the given index
    dicts_df = pd.read_pickle(os.path.join(path, obiFiles[idx]))

    # Compute the median of the DataFrame along axis=1
    median = dicts_df.median(axis=1)

    # Extract the symbol name from the file name
    symbol = obiFiles[idx].split("_")[0]

    # Assign the median values to the variable 'data'
    data = median

    # Fit the power-law model to the data
    fit = powerlaw.Fit(data, discrete=True)
    #
    # # Plot PDF and CCDF
    # fig_pdf = plt.figure()
    # ax_pdf = fig_pdf.add_subplot(1, 1, 1)
    # fit.plot_pdf(ax=ax_pdf, color='b', linewidth=2)
    # fit.power_law.plot_pdf(ax=ax_pdf, color='b', linestyle='--')
    # ax_pdf.set_ylabel(u"p(X),  p(X≥x)")
    # ax_pdf.set_xlabel(r"OBI Values ")
    # fig_pdf.savefig(os.path.join(powerLawFiguresLocation, f'{symbol}_pdf.png'), dpi=300)
    # fig_ccdf = plt.figure()
    # ax_ccdf = fig_ccdf.add_subplot(1, 1, 1)
    # FigCCDFmax = fit.plot_ccdf(ax=ax_ccdf, color='g', linewidth=2)
    # fit.power_law.plot_ccdf(ax=ax_ccdf, color='g', linestyle='--')
    # ax_ccdf.set_ylabel(u"p(X≥x)", fontsize=16, fontname='Times New Roman')
    # ax_ccdf.set_xlabel(r"OBI Values ", fontsize=16, fontname='Times New Roman')
    # plt.xticks(fontsize=12, fontname='Times New Roman')
    # plt.yticks(fontsize=12, fontname='Times New Roman')
    # handles, labels = ax_ccdf.get_legend_handles_labels()
    # leg = ax_ccdf.legend(handles, labels, loc=3)
    # leg.draw_frame(False)
    # fig_ccdf.savefig(os.path.join(powerLawFiguresLocation, f'{symbol}_ccdf.png'), dpi=300)
    #
    # # Assuming 'data' is the median of your DataFrame as calculated previously
    # data = median  # median calculated from your DataFrame
    # ########################################################################
    # # Initialize the power law fit with no xmax
    # fit_no_xmax = powerlaw.Fit(data, discrete=True, xmax=None)
    #
    # # Plot CCDF for the fit with no xmax
    # FigCCDFmax = fit_no_xmax.plot_ccdf(color='b', label=r"Empirical, no $x_{max}$")
    # fit_no_xmax.power_law.plot_ccdf(color='b', linestyle='--', ax=FigCCDFmax, label=r"Fit, no $x_{max}$")
    # plt.xticks(fontsize=12, fontname='Times New Roman')
    # plt.yticks(fontsize=12, fontname='Times New Roman')
    #
    # # Initialize the power law fit with xmax set to 1000
    # fit_with_xmax = powerlaw.Fit(data, discrete=True, xmax=1000)
    #
    # # Set labels and legend for the plot
    # FigCCDFmax.set_ylabel(u"p(X≥x)", fontsize=16, fontname='Times New Roman')
    # FigCCDFmax.set_xlabel(r"OBI Values", fontsize=16, fontname='Times New Roman')
    # handles, labels = FigCCDFmax.get_legend_handles_labels()
    # leg = FigCCDFmax.legend(handles, labels, loc=3)
    # plt.xticks(fontsize=12, fontname='Times New Roman')
    # plt.yticks(fontsize=12, fontname='Times New Roman')
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # leg.draw_frame(False)
    #
    # # Save the plot to a file
    # ccdf_plot_file = os.path.join(powerLawFiguresLocation, f'{symbol}_ccdf_comparison.png')
    # plt.savefig(ccdf_plot_file, dpi=300)
    # # saving more plots
    # # Power-law fitting and plotting
    # # fit_no_xmax = powerlaw.Fit(data, discrete=True, xmax=None)
    # # FigCCDFmax = fit_no_xmax.plot_ccdf(color='b', label=r"Empirical, no $x_{max}$")
    # # fit_no_xmax.power_law.plot_ccdf(color='b', linestyle='--', ax=FigCCDFmax, label=r"Fit, no $x_{max}$")
    #
    # fit_with_xmax = powerlaw.Fit(data, discrete=True, xmax=1000)
    # fit_with_xmax.plot_ccdf(color='r', label=r"Empirical, $x_{max}=1000$")
    # fit_with_xmax.power_law.plot_ccdf(color='r', linestyle='--', ax=FigCCDFmax, label=r"Fit, $x_{max}=1000$")
    # plt.xticks(fontsize=12, fontname='Times New Roman')
    # plt.yticks(fontsize=12, fontname='Times New Roman')
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    #
    # # Set labels and legend for the plot
    # FigCCDFmax.set_ylabel(u"p(X≥x)")
    # FigCCDFmax.set_xlabel(r"OBI Values")
    # plt.xticks(fontsize=12, fontname='Times New Roman')
    # plt.yticks(fontsize=12, fontname='Times New Roman')
    # handles, labels = FigCCDFmax.get_legend_handles_labels()
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # leg = FigCCDFmax.legend(handles, labels, loc=3)
    # leg.draw_frame(False)
    #
    # # Save the plot to a file
    # ccdf_plot_file = os.path.join(powerLawFiguresLocation, f'{symbol}_ccdf_additional_comparison.png')
    # plt.savefig(ccdf_plot_file, dpi=300)
    #
    # # Compare distributions
    # R, p = fit.distribution_compare('power_law', 'lognormal')
    #
    # # Plot CCDFs of empirical data and both fits
    # # Create figure
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    #
    # # Plot CCDFs
    # fit.plot_ccdf(ax=ax, linewidth=3, label='Empirical Data')
    # fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--', label='Power law fit')
    # fit.lognormal.plot_ccdf(ax=ax, color='g', linestyle='--', label='Lognormal fit')
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    #
    # ax.legend()
    # plt.tight_layout()
    #
    # # Save figure
    # fig.savefig(os.path.join(powerLawFiguresLocation,f'{symbol}_fitted_distro_ccdf_plot.png'), dpi=300)
    #
    # # Set labels and legend
    # fig.set_ylabel(u"p(X≥x)")
    # fig.set_xlabel("OBI Values")
    # handles, labels = fig.get_legend_handles_labels()
    # fig.legend(handles, labels, loc=3)
    #
    # # Save the plot
    # figname = os.path.join(powerLawFiguresLocation, f'{symbol}_FigLognormal')
    # fig.savefig(f'{figname}.png', bbox_inches='tight')

    # Save the results of the power-law fit
    results = {
        'alpha': fit.power_law.alpha,
        'sigma': fit.power_law.sigma,
        'xmin': fit.xmin,
        'xmax': fit.xmax,
        'xmin_distance': fit.xmin_distance,
        'alpha_std_err': fit.power_law.sigma,
        'distribution': fit.power_law.name,
        'D': fit.D,
        'n': fit.n,
        'n_tail': fit.n_tail,
        'loglikelihood': fit.power_law.loglikelihood,
        'fitted model':fit
    }

    # Save the fitted model
    fitted_model_title = os.path.join(powerLawResults, f'powerLaw_{symbol}_fitted_model.pkl')
    with open(fitted_model_title, 'wb') as f:
        pickle.dump(results, f)

    # Return the index and symbol for confirmation
    return idx, symbol


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, idx) for idx in range(14)]  # Process indices 0 to 13
        for future in concurrent.futures.as_completed(futures):
            idx, symbol = future.result()
            print(f"Completed processing for index {idx}, symbol {symbol}")

