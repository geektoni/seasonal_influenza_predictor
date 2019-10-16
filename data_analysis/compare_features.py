# -*- coding: utf-8 -*-

"""Script which can be used to compare the features obtained of two different influenza models

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

def get_results_filename(basepath, country):
    files = [f for f in glob.glob(basepath + "/*_information_{}.csv".format(country), recursive=True)]
    season_years = os.path.basename(files[0]).split("_")[0]
    return season_years

if __name__ == "__main__":

    # Read the command line arguments
    args = docopt(__doc__)

    # Read some config variables
    base_dir = args["--basedir"] if args["--basedir"] else "../models/complete_results"
    country = args["--country"] if args["--country"] else "italy"
    future = "no-future" if args["--no-future"] else "future"

    # Get keywords coming from the various methods
    print("")
    keywords_standard = pd.read_csv(os.path.join("../data/keywords", "keywords_{}.txt".format(country)), header=None, names=["page_name"])
    print("Standard keywords Size: {}".format(len(keywords_standard)))

    # Get keywords coming from the various methods
    keywords_cyclerank = pd.read_csv(os.path.join("../data/keywords", "keywords_cyclerank_{}.txt".format(country)), header=None, names=["page_name"])
    print("Cyclerank keywords Size: {}".format(len(keywords_cyclerank)))

    common_keywords = set.intersection(set(keywords_standard.page_name), set(keywords_cyclerank.page_name))
    print("Common keywords Size: {}, {}, {}".format(len(common_keywords), len(common_keywords)/len(keywords_standard), len(common_keywords)/len(keywords_cyclerank)))
    print("")

    # Read the baseline results and merge them
    baseline_results_path= os.path.join(base_dir, args["<baseline>"], future, country)

    season_years = get_results_filename(baseline_results_path, country)
    season_years_baseline = season_years
    baseline_result_file = os.path.join(baseline_results_path, "{}_features_{}.csv".format(season_years, country))
    baseline_results_df = pd.read_csv(baseline_result_file)[["season", "page_name", "value"]]

    # Concat all the other results
    other_results_df = None
    for other_results in args["<other_method>"]:
        other_results_path = baseline_results_path= os.path.join(base_dir, other_results, future, country)
        season_years = get_results_filename(baseline_results_path, country)
        other_result_file = os.path.join(baseline_results_path, "{}_features_{}.csv".format(season_years, country))

        if other_results_df is None:
            other_results_df = pd.read_csv(other_result_file)[["page_name", "value"]]
            other_results_df = other_results_df.rename(columns={"page_name": "page_name_{}".format(other_results), "value":"value_{}".format(other_results)})
        else:
            current_other_results_df = pd.read_csv(other_result_file)[["page_name", "value"]]
            current_other_results_df = current_other_results_df.rename(columns={"page_name": "page_name_{}".format(other_results), "value":"value_{}".format(other_results)})
            other_results_df = pd.concat([other_results_df, current_other_results_df], axis=1)

    # Total results
    results = pd.concat([baseline_results_df, other_results_df], axis=1)

    # Count uniques years
    unique_years = results.season.unique()

    # Loop over the seasons and extract common pages
    total_common = []
    for y in unique_years:
        selected = results[results.season == y]
        for m in args["<other_method>"]:
            common = selected[selected.page_name == selected["page_name_{}".format(m)]]
            total_common += list(common.page_name.unique())
            print("{}, {} -> Common Pages = {}".format(y, m, len(common.page_name.unique())))

            how_many_in_common_keywords = set.intersection(set(selected.page_name), common_keywords)
            print("{}, {} -> Common Keywords = {}".format(y, m, len(how_many_in_common_keywords)))
            print("")

    # Print overall common features used:
    print("Total common pages used by the models")
    print(set(total_common))
    print("")


    #print(results)
