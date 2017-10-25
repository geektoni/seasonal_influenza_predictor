#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Machine learning model which uses Wikipedia data to predicts ILI levels.
Usage:
  model.py <year_start> <year_end> [--p]

Options:
  -p, --poisson Use the poisson model + LASSO
  -h, --help
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from colour import Color
from docopt import docopt
import csv

from models_utils import generate, generate_labels, generate_labels_one_year, generate_one_year, standardize_data, generate_keywords, stz, get_important_pages

from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict
from cvglmnet import cvglmnet

#### INITIALIZATION #####

# Parse the command line
arguments = docopt(__doc__)

# Year selected
year_start = int(arguments["<year_start>"])
year_end = int(arguments["<year_end>"])+1

# Model type
using_poisson = False
if (arguments["--p"]):
    using_poisson = True

# Selected columns that will be extracted from the dataframes
selected_columns = generate_keywords()

##### ALGORITHM ######

all_features_values = pd.DataFrame()
all_predicted_values = []
all_true_labels = pd.DataFrame()
total_weeks = []
all_weighted_feature=dict()

year_sel = (year_start, year_end)

for year_selected in range(year_sel[0], year_sel[1]):

    print("[*] ", year_selected )
    # Data generation from data files
    dataset = generate(year_selected)[selected_columns]
    labels = generate_labels(year_selected)
    data = generate_one_year(year_selected)[selected_columns]
    labels_test = generate_labels_one_year(year_selected)
    weeks = generate_labels_one_year(year_selected)["week"]

    # Create a copy of the dataset where we fills NaN values
    # with 0.
    dataset_zero = dataset.fillna(0)
    data_zero = data.fillna(0)

    # Standardize data
    train, test = standardize_data(dataset_zero, data_zero)

    # Regenerate panda dataframe with standardized data
    data_graph = pd.DataFrame(data=test, columns=selected_columns)

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


# Get important pages to generate the plot we need
important_pages = get_important_pages(all_weighted_feature, 5)

# Print MSE of the two models
print("------------")
print("LASSO XVal  General MSE: ", mean_squared_error(all_true_labels["incidence"].fillna(0), all_predicted_values))
print("------------")

# Print Pearson Coefficient
print("------------")
print(np.corrcoef(all_predicted_values, all_true_labels["incidence"].fillna(0), rowvar=False))
print("------------")

# Plot some informations
fig = plt.figure(3, figsize=(15, 6))
ax = fig.add_subplot(111)

plt.title("Stagioni Influenzali "+str(year_sel[0]-1)+" - "+str(year_sel[1]-1), fontsize=18)
plt.ylabel("Incidenza su 100000 persone", fontsize=17)
plt.xlabel("Anno-Settimana", fontsize=17)

weeks_used = []
for k, v in enumerate(total_weeks):
    if k%2==0:
        weeks_used.append(v)
    else:
        weeks_used.append(" ")

plt.xticks(range(0, len(weeks_used)), weeks_used, rotation="vertical", fontsize=15)
plt.yticks(fontsize=15)

plt.plot(range(0, len(all_predicted_values)), all_predicted_values, 'r-', label="Modello Lineare", linewidth=3)
plt.plot(range(0, len(all_true_labels["incidence"])), all_true_labels["incidence"].fillna(0), 'k-', label="InfluNet", linewidth=3)

# Add dotted line into the graph to delimit the influenza seasons.
for i in range(1, (year_sel[1]-year_sel[0])):
    plt.axvline(26*i, linestyle="dashed", color="k", linewidth=3)

# Update some graphical options
plt.grid()
plt.legend(fontsize=16, loc="upper right")
plt.tight_layout()

# Save the grapg on file
plt.savefig("appendix_linear_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+".png", dpi=150)

# Feature plot
figf = plt.figure(4, figsize=(15, 6))
axf = fig.add_subplot(111)

plt.title("Variazione pageview "+str(year_sel[0]-1)+" - "+str(year_sel[1]-1), fontsize=18)
plt.ylabel("Numero Pageview", fontsize=17)
plt.xlabel("Anno-Settimana", fontsize=17)

plt.xticks(range(0, len(weeks_used)), weeks_used, rotation="vertical", fontsize=15)
plt.yticks(fontsize=15)

std_all_features_values = stz(all_features_values)
for key, value in important_pages:
 plt.plot(range(0, len(std_all_features_values[key])), std_all_features_values[key], '-', label=key, linewidth=3)
plt.plot(range(0, len(all_true_labels["incidence"])), stz(all_true_labels["incidence"]), 'k-', label="InfluNet", linewidth=3)

plt.grid()
plt.legend(fontsize=16, loc="best")
plt.tight_layout()

plt.savefig("appendix_linear_feature_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+".png", dpi=150)

# Write information to file
#with open(str(year_selected)+"_information.csv", 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    spamwriter.writerow(['mse', mean_squared_error(labels_test, result)])
#    spamwriter.writerow(['alpha', mean_squared_error(labels_test, result)])
#    for p in important_pages:
#        spamwriter.writerow([p[0], p[1]])