#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine learning model which uses Wikipedia data to predicts ILI levels.

Usage:
  model.py <year_start> <year_end> <dataset_path> <incidence_path> <keywords_file> [--p] [--f] [--v] [--d=<directory>] [--no-future]

  <year_start>      The first influenza season we want to predict.
  <year_end>        The last influenza season we want to predict.
  <dataset_path>    The path to dataset files (the ones which contain Wikipedia pageviews).
  <incidence_path>  The path to the files which specify the ILI levels for each week.
  <keywords_file>   The path to the file which contains the Wikipedia pages used.
  -p, --poisson     Use the Poisson model + LASSO instead of the linear one.
  -f, --file        Write informations to file
  -v, --verbose     Output more informations
  -d, --directory   Select a directory in which save your files
  -n, --no-future   Use a different method to train the model (avoid using seasonal influenza which are
                    later than the one we want to predict)
  -h, --help        Print this help message
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from docopt import docopt

from models_utils import generate, generate_labels, generate_labels_one_year, generate_one_year, standardize_data, generate_keywords, stz, get_important_pages, correlation_matrix

from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict
from cvglmnet import cvglmnet

#########################
#### INITIALIZATION #####
#########################

# Parse the command line
arguments = docopt(__doc__)

# Directory where to save files
save_directory="./"
if arguments["--d"]:
    save_directory = arguments["--d"]

# Year selected
year_start = int(arguments["<year_start>"])
year_end = int(arguments["<year_end>"])+1

# Model type
model_type = "Linear Model"
using_poisson = False
if arguments["--p"]:
    using_poisson = True
    model_type="Poisson Model"

# Feature path and labels
path_features = arguments["<dataset_path>"]
path_labels = arguments["<incidence_path>"]
keywords = arguments["<keywords_file>"]

# Selected columns that will be extracted from the dataframes
selected_columns = generate_keywords(keywords)

all_features_values = pd.DataFrame()
all_predicted_values = []
all_true_labels = pd.DataFrame()
total_weeks = []
all_weighted_feature=dict()

# The pair start-end seasons
year_sel = (year_start, year_end)

######################
##### ALGORITHM ######
######################

# For each season, train a model with the complete dataset (without the season)
# and predict the ILI incidence.
for year_selected in range(year_sel[0], year_sel[1]):

    print("------------")
    print("[*] ", year_selected )

    # Generate the list of excluded years
    excluded=[]
    if arguments["--no-future"]:
        for i in range(2007, 2018):
            if i>year_selected:
                excluded.append(i)

    # Data generation from data files
    dataset = generate(year_selected, excluded, path_features)[selected_columns]
    labels = generate_labels(year_selected, excluded, path_labels)
    data = generate_one_year(year_selected, path_features)[selected_columns]
    labels_test = generate_labels_one_year(year_selected, path_labels)
    weeks = labels_test["week"]

    # Create a copy of the dataset where we fills NaN values
    # with 0.
    dataset_zero = dataset.fillna(0)
    data_zero = data.fillna(0)

    # Standardize data
    train, test = standardize_data(dataset_zero, data_zero)

    # Create a Lasso Cross-Validation instance which will be
    # trained on the dataset in which NaN values are replaced
    # with the column mean.
    if not using_poisson:
        model = LassoCV(max_iter=100000, n_jobs=-1, n_alphas=1000, random_state=1)
        model.fit(train, labels["incidence"].fillna(0))
        result = model.predict(test)
    else:
        model = cvglmnet(x=train.as_matrix().copy(), y=labels["incidence"].fillna(0).copy().as_matrix(), family='poisson', alpha=1.0,
                       ptype="mse", parallel=True, nfolds=10)
        result = cvglmnetPredict(model, test.as_matrix(), ptype='response', s="lambda_min")
        coeff = cvglmnetCoef(model, s="lambda_min")

    # Get the feature coefficients
    if not using_poisson:
        coeff = model.coef_
    else:
        coeff = cvglmnetCoef(model, s="lambda_min")

    # Add the pair (value, coeff) to a dictionary
    for i in list(zip(coeff, selected_columns)):
        if (i[0] != 0):
            if all_weighted_feature.get(i[1], None) is None:
                all_weighted_feature[i[1]] = [i[0]]
            else:
                all_weighted_feature[i[1]].append(i[0])

    # Add the value to global variables
    all_features_values= all_features_values.append(data.fillna(0))
    all_predicted_values.extend(result)
    all_true_labels = all_true_labels.append(labels_test)
    total_weeks.extend(weeks)

    # If we are verbose print more informations
    if arguments["--v"]:
        index_peak = labels_test['incidence'].fillna(0).idxmax(axis=1)
        index_peak_m = np.argmax(result)
        print("[*] Influenza Season Peak (Week):", labels_test.iloc[index_peak]["week"])
        print("[*] Influenza Seasons Predicted Peak (Week): ", labels_test.iloc[index_peak_m]["week"])
        print("[*] Influenza Season Peak Value: ", labels_test.iloc[index_peak]["incidence"])
        print("[*] Influenza Season Predicted Peak Value: ", result[index_peak_m])
        print("------------")

# Get important pages to generate the plot we need
important_pages = get_important_pages(all_weighted_feature, 5)

# Print MSE of the two models
mse = mean_squared_error(all_true_labels["incidence"].fillna(0), all_predicted_values)
print("------------")
print("MSE: ", mse)
print("------------")

# Print Pearson Coefficient
pcc = np.corrcoef(all_predicted_values, all_true_labels["incidence"].fillna(0), rowvar=False)[0][1]
print("------------")
print("Pearson Correlation Coeff: ", pcc)
print("------------")

# Print important pages
print(tabulate(important_pages, headers=["Page name", "Weight"]))

###########################
#### GRAPHS GENERATION ####
###########################

# Plot some informations
fig = plt.figure(3, figsize=(15, 6))
ax = fig.add_subplot(111)

# Set up the graph title, x and y axis titles
plt.title("Influenza Seasons "+str(year_sel[0]-1)+" - "+str(year_sel[1]-1), fontsize=18)
plt.ylabel("Incidence on 100000 people", fontsize=17)
plt.xlabel("Year-Week", fontsize=17)

# Generate the x axis labels in such way to
# have the year-week pair only every two ticks.
weeks_used = []
for k, v in enumerate(total_weeks):
    if k%2==0:
        weeks_used.append(v)
    else:
        weeks_used.append(" ")

# Set up the axes ticks
plt.xticks(range(0, len(weeks_used)), weeks_used, rotation="vertical", fontsize=15)
plt.yticks(fontsize=15)

# Plot the model result and the incidence
plt.plot(range(0, len(all_predicted_values)), all_predicted_values, 'r-', label=model_type, linewidth=3)
plt.plot(range(0, len(all_true_labels["incidence"])), all_true_labels["incidence"].fillna(0), 'k-', label="Incidence", linewidth=3)

# Add dotted line into the graph to delimit the influenza seasons.
for i in range(1, (year_sel[1]-year_sel[0])):
    plt.axvline(26*i, linestyle="dashed", color="k", linewidth=3)

# Update some graphical options
plt.grid()
plt.legend(fontsize=16, loc="upper right")
plt.tight_layout()

# Save the graph on file
plt.savefig(save_directory+"season_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+".png", dpi=150)

# Feature plot
figf = plt.figure(4, figsize=(15, 6))
axf = fig.add_subplot(111)

# Generate the graph showing the pageview variation of the top 5 Wikipedia's pages
# choosen by the model.
plt.title("Pageview variation "+str(year_sel[0]-1)+" - "+str(year_sel[1]-1), fontsize=18)
plt.ylabel("Pageview value", fontsize=17)
plt.xlabel("Year-Week", fontsize=17)

# Plot the axes labels
plt.xticks(range(0, len(weeks_used)), weeks_used, rotation="vertical", fontsize=15)
plt.yticks(fontsize=15)

# Plot all the pageview data and the ILI incidence
std_all_features_values = stz(all_features_values)
for key, value in important_pages:
 plt.plot(range(0, len(std_all_features_values[key])), std_all_features_values[key], '-', label=key, linewidth=3)
plt.plot(range(0, len(all_true_labels["incidence"])), stz(all_true_labels["incidence"]), 'k-', label="Incidence", linewidth=3)

# Add dotted line into the graph to delimit the influenza seasons.
for i in range(1, (year_sel[1]-year_sel[0])):
    plt.axvline(26*i, linestyle="dashed", color="k", linewidth=3)

# Update some graphical options
plt.grid()
plt.legend(fontsize=16, loc="best")
plt.tight_layout()

# Save again the graph on file
plt.savefig(save_directory+"season_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+"_features.png", dpi=150)

# If --f then we write some information to file
if arguments["--f"]:
    with open(save_directory+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+"_information.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['mse', mean_squared_error(all_true_labels["incidence"].fillna(0), all_predicted_values)])
        for p in important_pages:
            spamwriter.writerow([p[0], float(p[1])])

# Correlation Matrix
new_pages=[]
for p in important_pages:
    new_pages.append(p[0])
first = pd.DataFrame(all_features_values[new_pages])
second = pd.DataFrame(all_true_labels["incidence"]).set_index(first.index.values)
first['incidence'] = second
new_pages.append("Incidence")
correlation_matrix(first, "Correlation Matrix "+str(year_sel[0]-1)+"-"+str(year_sel[1]-1), new_pages, save_directory+"cmatrix_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+".png")