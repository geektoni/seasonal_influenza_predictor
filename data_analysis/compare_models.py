# -*- coding: utf-8 -*-

"""Script which can be used to compare the results of two different influenza model

Usage:
  compare_models.py <baseline> <other_method>... [--country=<country_name>] [--no-future] [--basedir=<directory>] [--start-year=<start_year>] [--end-year=<end_year>] [--save] [--no-graph]

  <baseline>      Data file of the first model
  <other_method>     Data file of the second model
  -h, --help        Print this help message
"""

import os
import glob

from docopt import docopt
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def correct_name(value):
    if value == "new_data" or value == "old_data":
        return "LASSO (Categories)"
    elif value == "cyclerank":
        return "LASSO (Cyclerank)"
    elif value == "pageviews":
        return "LASSO (Categories)"
    elif value == "cyclerank_pageviews":
        return "LASSO (Cyclerank)"


def get_results_filename(basepath, country):
    files = [f for f in glob.glob(basepath + "/*_information_{}.csv".format(country), recursive=True)]
    season_years = os.path.basename(files[0]).split("_")[0]
    return season_years

if __name__ == "__main__":

    # Read the command line arguments
    args = docopt(__doc__)

    # Read some config variables
    base_dir = args["--basedir"] if args["--basedir"] else "../complete_results"
    country = args["--country"] if args["--country"] else "italy"
    future = "no-future" if args["--no-future"] else "future"

    # Read the baseline results and merge them
    baseline_results_path= os.path.join(base_dir, args["<baseline>"], future, country)

    season_years = get_results_filename(baseline_results_path, country)
    season_years_baseline = season_years
    baseline_result_file = os.path.join(baseline_results_path, "{}_information_{}.csv".format(season_years, country))
    baseline_results_df = pd.read_csv(baseline_result_file).rename(columns={"mse": "mse_{}".format(args["<baseline>"]), "pcc":"pcc_{}".format(args["<baseline>"])})

    # Concat all the other results
    other_results_df = None
    for other_results in args["<other_method>"]:
        other_results_path = baseline_results_path= os.path.join(base_dir, other_results, future, country)
        season_years = get_results_filename(baseline_results_path, country)
        other_result_file = os.path.join(baseline_results_path, "{}_information_{}.csv".format(season_years, country))

        if other_results_df is None:
            other_results_df = pd.read_csv(other_result_file)
            other_results_df = other_results_df.rename(columns={"mse": "mse_{}".format(other_results), "pcc":"pcc_{}".format(other_results)})
        else:
            current_other_results_df = pd.read_csv(other_result_file)
            current_other_results_df = current_other_results_df.rename(columns={"mse": "mse_{}".format(other_results), "pcc":"pcc_{}".format(other_results)})
            other_results_df = pd.merge(other_results_df, current_other_results_df, on="season", how="outer")

    # Total results
    results = pd.merge(baseline_results_df, other_results_df, on="season", how="outer")

    # FIXME: There may be season for which the mse and pcc are NaN (because maybe we are
    # comparing new_data with old_data). Those lines must be written as nan in the final result.
    for other_results in args["<other_method>"]:
        #results["imp_mse_{}".format(other_results)] = np.where((results["mse"] - results["mse_{}".format(other_results)])>=0, 'yes', 'no')
        #results["imp_pcc_{}".format(other_results)] = np.where((results["pcc"] - results["pcc_{}".format(other_results)])>=0, 'no', 'yes')
        results["ΔMSE ({})".format(other_results)] = results["mse_{}".format(args["<baseline>"])] - results["mse_{}".format(other_results)]
        results["ΔPCC ({})".format(other_results)] = results["pcc_{}".format(args["<baseline>"])] - results["pcc_{}".format(other_results)]


    # Specify which columns we want to obtain from the dataframe
    baseline_name = "pagecounts+pageviews" if args["<baseline>"] == "new_data" else "pageviews"
    printable_columns = []
    printable_columns.append("season")
    printable_columns.append("MSE ({})".format(baseline_name))
    printable_columns.append("PCC ({})".format(baseline_name))
    printable_columns += ["ΔMSE ({})".format(m) for m in args["<other_method>"]]
    printable_columns += ["ΔPCC ({})".format(m) for m in args["<other_method>"]]


    results = results.rename(columns={
    "mse_{}".format(args["<baseline>"]): "MSE ({})".format(baseline_name),
    "pcc_{}".format(args["<baseline>"]): "PCC ({})".format(baseline_name)})

    print(results[printable_columns])

    #### GENERATE THE GRAPH
    baseline_prediction_path = os.path.join(base_dir, args["<baseline>"], future, country)
    season_years = get_results_filename(baseline_prediction_path, country)
    baseline_prediction_file = os.path.join(baseline_prediction_path, "{}-prediction.csv".format(season_years))
    baseline_prediction_df = pd.read_csv(baseline_prediction_file)[["prediction", "week", "incidence"]].rename(columns={"prediction": "prediction_{}".format(args["<baseline>"])})

    # Concat all the other results
    other_prediction_df = None
    for other_result in args["<other_method>"]:
        other_prediction_path = os.path.join(base_dir, other_result, future, country)
        season_years = get_results_filename(other_prediction_path, country)
        other_prediction_file = os.path.join(other_prediction_path, "{}-prediction.csv".format(season_years))

        if other_prediction_df is None:
            other_prediction_df = pd.read_csv(other_prediction_file).drop(["incidence"], axis=1)[["prediction", "week"]]
            other_prediction_df = other_prediction_df.rename(columns={"prediction": "prediction_{}".format(other_result)})
        else:
            current_other_prediction_df = pd.read_csv(other_prediction_file).drop(["incidence"], axis=1)[["prediction", "week"]]
            current_other_prediction_df = current_other_prediction_df.rename(columns={"prediction": "prediction_{}".format(other_result)})
            other_prediction_df = pd.merge(other_prediction_df, current_other_prediction_df, on="week", how="outer")

    # Total results
    prediction_results = pd.merge(baseline_prediction_df, other_prediction_df, on="week", how="outer")
    prediction_results = prediction_results.dropna()

    # Get only the weeks we want
    start_year = season_years_baseline.split("-")[0]+"-42" if not args["--start-year"] else args["--start-year"]
    end_year = season_years_baseline.split("-")[1]+"-15" if not args["--end-year"] else args["--end-year"]
    start_season = prediction_results["week"] >= start_year
    end_season = prediction_results["week"] <= str(int(end_year.split("-")[0])+1)+"-"+end_year.split("-")[1]
    total = start_season & end_season

    prediction_results = prediction_results[total]
    prediction_results = prediction_results.reset_index()
    prediction_results = prediction_results.drop(["index"], axis=1)
    step = int(len(prediction_results)*0.05)
    if (step ==0):
        step=1

    # Get max y value
    min_values = prediction_results.min()
    max_y_value = int(prediction_results.drop(["week"], axis=1).max().max())
    min_y_value = int(prediction_results.drop(["week"], axis=1).min().min())
    step_y = int(max_y_value*0.05) if int(max_y_value*0.05) != 0 else 1

    index=1
    lines = []
    labels = []
    all_methods = [args["<baseline>"]]+args["<other_method>"]
    fig = plt.figure(figsize=(11,6))
    #palette = sns.color_palette("Paired")
    palette = ["red", "blue", "green", "yellow", "green"]
    end_of_seasons = [i for i, n in enumerate(prediction_results["week"].to_list()) if n.split("-")[1] == "15"]
    for other_result in all_methods:
        plt.subplot(len(all_methods), 1, index)
        prediction_results_plot = prediction_results.drop(["week"], axis=1)[["incidence", "prediction_{}".format(other_result)]]
        ax = sns.lineplot(data=prediction_results_plot, style="event", dashes=False, palette=["black", palette[index-1]], legend=False)
        lines.append(ax.get_lines()[1])
        labels.append("{}".format(correct_name(other_result)))
        index+=1
        ax.set_xticks([i for i in np.arange(len(prediction_results["week"]), step=step)])
        ax.set_xticklabels([" " for i in np.arange(len(prediction_results["week"]), step=step)])
        plt.ylabel("ILI Incidence",  fontsize=11)
        for i in end_of_seasons:
            plt.axvline(x=i, color='k', linestyle='--')
        plt.ylim(min_y_value-2, max_y_value+2)

    lines.append(ax.get_lines()[0])
    labels.append("ILI Incidence")

    plt.xticks(np.arange(len(prediction_results["week"]), step=step), prediction_results["week"].iloc[::step], rotation=90, fontsize=9)
    plt.xlabel("Year-Week", fontsize=11)
    plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3, fontsize=11)

    if not args["--no-graph"]:
        fig.tight_layout()
        plt.show()
    else:
        save_filename = "{}_{}_compare_results_{}_{}.png".format(start_year, end_year, args["<baseline>"], country)
        plt.savefig(save_filename, dpi=200, bbox_inches='tight')

        save_filename = "{}_{}_compare_results_{}_{}.csv".format(start_year, end_year, args["<baseline>"], country)
        results[printable_columns].to_csv(os.path.join(".", save_filename), index=False, index_label=False)
