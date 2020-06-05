import argparse
import pandas as pd
import numpy as np

import os
import glob

import seaborn as sns
sns.set(context="paper", font_scale=1.3)

import matplotlib.pyplot as plt

def get_results_filename(basepath, country):
    files = [f for f in glob.glob(basepath + "/*_information_{}.csv".format(country), recursive=True)]
    season_years = os.path.basename(files[0]).split("_")[0]
    return season_years

def get_keywords_prefix(model):
    """
    Return the correct keyword's file prefix given the model
    :param model: name of the model
    :return: keyword's file prefix
    """
    if model == "cyclerank" or model == "cyclerank_pageviews":
        return "keywords_cyclerank"
    elif model == "pagerank" or model == "pagerank_pageviews":
        return "keywords_pagerank"
    else:
        return "keywords"

def convert_naming(model):
    if model == "cyclerank":
        return "CycleRank (PV+PC)"
    elif model == "cyclerank_pageviews":
        return "CycleRank (PV)"
    elif model == "pagerank":
        return "PageRank (PV+PC)"
    elif model == "pagerank_pageviews":
        return "PageRank (PV)"
    elif model == "pageviews":
        return "Categories (PV)"
    else:
        return "Categories (PV+PC)"


def get_intersection_dataframe(original_data, percentage=False):

    # Compute the intersection between all of them
    intersection_df = []
    intersection_columns = ["index"]
    for first in original_data:
        partial_value = [convert_naming(first)]
        for second in original_data:
            intersection_value = set.intersection(set(original_data[first].page_name),
                                                  set(original_data[second].page_name))

            if percentage:
                partial_value.append((len(intersection_value)/len(original_data[first]))*100)
            else:
                partial_value.append(len(intersection_value))
        intersection_df.append(partial_value)

    # Convert the intersection into a dataframe
    intersection_columns = ["index"] + [convert_naming(x) for x in list(original_data.keys())]
    intersection_df = pd.DataFrame(intersection_df, columns=intersection_columns)
    intersection_df = intersection_df.set_index("index")
    return intersection_df


def get_min_max_mean_feature_selected(df, start_year=2015, end_year=2019):
    min = np.inf
    max = 0
    mean = 0
    seasons = df.season.unique()
    correct_seasons = 0
    for s in seasons:
        start_end = s.split("-")
        if int(start_end[0]) >= start_year and int(start_end[1]) <= end_year:
            correct_seasons += 1
            features_slice = df[df.season == s]
            features_selected = len(features_slice)

            if features_selected > max:
                max = features_selected
            if features_selected < min:
                min = features_selected

            mean += features_selected

    return min, max, mean/correct_seasons

def get_feature_dictionary(df, start_year=2015, end_year=2019):
    """
    Group all the features into a dictionary. Namely, we generate for each
    feature a list of values which indicates their weight in each of the models
    under scrutiny
    :param df: the feature dataframe
    :param start_year: start year to consider
    :param end_year: end year to consider
    :return: a dictionary
    """

    feature_dictionary = {}
    seasons = df.season.unique()
    for s in seasons:
        start_end = s.split("-")
        if int(start_end[0]) >= start_year and int(start_end[1]) <= end_year:
            features_slice = df[df.season == s]
            for index, row in features_slice.iterrows():
                if row["page_name"] in feature_dictionary:
                    feature_dictionary[row["page_name"]].append(row["value"])
                else:
                    feature_dictionary[row["page_name"]] = [row["value"]]
    return feature_dictionary


def get_important_pages(important_pages, top=10, influenza_seasons=4):
    """
    Get the most important feature selected by the model.

    :param important_pages: a dictionary with, for each of the features,
    a list of their weights in each of the models.
    :param top: how many feature we want to return.
    :return: the top feature
    """
    imp_pages_avg = dict((k, sum(v) / float(influenza_seasons)) for k, v in important_pages.items())
    _terms_avg_top = sorted(sorted(imp_pages_avg.items(),
                                   key=lambda value: value[0]),
                            key=lambda value: value[1],
                            reverse=True
                            )
    return _terms_avg_top[0:top]



if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("models", metavar="N", nargs="+", type=str, help="Models we want to check")
    parser.add_argument("--country", type=str, default="italy", help="Country name")
    parser.add_argument("--basedir", type=str, default="../complete_results")
    parser.add_argument("--future", default=True, action="store_false")
    parser.add_argument("--start-year", default=2015, type=int)
    parser.add_argument("--end-year", default=2019, type=int)
    parser.add_argument("--save", default=False, action="store_true")
    args = parser.parse_args()

    # Set some variables
    country = args.country
    base_dir = args.basedir
    future = "future" if args.future else "no-future"

    # Get the keywords for all the models, then compute the intersection between all of them
    total_keywords = {}
    total_features = {}
    for model_name in args.models:

        # Get models keywords
        keywords_prefix = get_keywords_prefix(model_name)
        model_total_keywords = pd.read_csv(os.path.join("../data/keywords",
                                                        "{}_{}.txt".format(keywords_prefix, country)),
                                           header=None, names=["page_name"])
        total_keywords[model_name] = model_total_keywords

        # Get selected features
        baseline_results_path = os.path.join(base_dir, model_name, future, country)
        season_years = get_results_filename(baseline_results_path, country)
        features_file = os.path.join(baseline_results_path, "{}_features_{}.csv".format(season_years, country))
        features = pd.read_csv(features_file)[["season", "page_name", "value"]]

        # Compute the best features over all models
        feature_dictionary = get_feature_dictionary(features)
        best_features = get_important_pages(feature_dictionary, top=300)
        best_features = [[x[0], x[1]] for x in best_features]

        total_features[model_name] = pd.DataFrame(best_features, columns=["page_name", "mean_weigth"])

        # Get min, max and mean features
        min, max, mean = get_min_max_mean_feature_selected(features)
        print("\t{}\t{}\t{}\t{:.2f}".format(convert_naming(model_name), min, max, mean))


    # Get percentage of common features
    sns.heatmap(get_intersection_dataframe(total_keywords, percentage=True), annot=True, fmt=".2f",
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75), linewidths=.5)
    if not args.save:
        plt.ylabel("")
        plt.title(country.title())
        plt.tight_layout()
        plt.show()

    # Find the common best feature between the various models
    intersection_features = get_intersection_dataframe(total_features, percentage=False)
    sns.heatmap(intersection_features, annot=True, fmt="d",
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75), linewidths=.5)

    # Save the top-5 feature for each model
    top_5_features = pd.DataFrame(columns=list(total_features.keys()))
    for k in total_features:
        top_5_features[k] = total_features[k]["page_name"][0:5].apply(lambda x: x.replace("_", " "))


    plt.ylabel("")
    plt.title(country.title())
    plt.tight_layout()

    if not args.save:
        plt.show()
    else:
        save_filename = "common_feature_{}.png".format(country)
        plt.savefig(save_filename, dpi=200, bbox_inches='tight')

        save_filename = "top_5_features_{}.csv".format(country)
        top_5_features.to_csv(save_filename, index=False)


