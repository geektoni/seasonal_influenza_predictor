#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LASSO model which uses wikipedia data to predicts ILI levels
Usage:
  generate_weekly_data.py <year>

Options:
  -h, --help
"""

import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from docopt import docopt
from colour import Color
import csv

from models_utils import generate, generate_labels, generate_labels_one_year, generate_one_year, cross_validation_glm

from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict

#### INITIALIZATION #####

# Parse the command line
arguments = docopt(__doc__)

# Year selected
year_selected = int(arguments["<year>"])

# Feature path and labels
path_features = "./../data/wikipedia"
path_labels = "./../data/influnet/csv"

# Selected columns that will be extracted from the dataframes
selected_columns = []
file_ = open("../data/keywords/keywords2.txt", "r")
for line in file_:
    if line != "Week":
        selected_columns.append(line.replace("\n", "").replace("\\", ""))

##### ALGORITHM ######

# Data generation from data files
dataset = generate(year_selected)[selected_columns]
labels = generate_labels(year_selected)["Incidenza Totale"]
data = generate_one_year(year_selected)[selected_columns]
labels_test = generate_labels_one_year(year_selected)["Incidenza Totale"]
weeks = generate_labels_one_year(year_selected)["Settimana"]

print ("Train set shape (rows/columns): ", dataset.shape)
print ("Test set shape (rows/columns): ", data.shape)
print ("------------")


# Create a copy of the dataset where we fills NaN values
# with 0.
dataset_zero = dataset.fillna(0);
data_zero = data.fillna(0);

# Standardize data
data_total = numpy.concatenate((dataset_zero, data_zero), axis=0)
dmean = data_total.mean(axis=0)
dmax = data_total.max(axis=0)
dmin = data_total.min(axis=0)
dmax_min = dmax-dmin
dataset_imp = (dataset_zero-dmean)/dmax_min
data_imp = (data_zero-dmean)/dmax_min
dataset_imp[numpy.isnan(dataset_imp)] = 0
data_imp[numpy.isnan(data_imp)] = 0

# Create the poisson model (use cross-validation to obtain the best alpha value
result = cross_validation_glm(dataset_imp.as_matrix(), labels, data_imp, labels_test, runs=50)

# Get the best fitted model
fit = result[0][1]

# Predict
result_lcv = cvglmnetPredict(fit, data_imp.as_matrix(), ptype = 'response', s = "lambda_min")

# Get the features weights
coeff = cvglmnetCoef(fit, s = "lambda_min")

# Regenerate panda dataframe with standardized data
data_graph = pd.DataFrame(data=data_imp, columns=selected_columns)

# Extract which features seems to be important
# from the Poisson model
important_pages=[]
graph_pages=[]
for i in list(zip(coeff, selected_columns)):
    if (i[0] != 0):
        if (i[0] > 0):
            graph_pages.append([i[1], i[0]])
        important_pages.append([i[1], i[0]])

# Sort the pages by weights
def getKey(item):
    return item[1]
important_pages = sorted(important_pages, key=getKey, reverse=True)
graph_pages = sorted(graph_pages, key=getKey, reverse=True)

# Print important pages
print ("------------")
print ("Pages which their weight is != from 0: ")
print (tabulate(important_pages, headers=["Page", "Mean"]))

# Prim MSE of the two models
print ("------------")
print ("Poisson MSE: ", mean_squared_error(labels_test, result_lcv))

# Plot some informations
plt.figure(figsize=(20, 10))
plt.ylabel("Incidenza su 1000 persone")
plt.xlabel("Settimane")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

plt.plot(range(0, len(result_lcv)), result_lcv, 'o-', label="Poisson Model")
plt.plot(range(0, len(labels_test)), labels_test, 'x-', label="Actual Value")

plt.legend()
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.savefig(str(year_selected)+"_poisson.png")

plt.figure(2, figsize=(20, 10))
plt.ylabel("Numero di visite")
plt.xlabel("Settimane")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

# All possible colors
linestyles = ['-', '--', '-.', ':']
colors =list(Color("blue").range_to(Color("red"), 10))
for i in range(0, 10):
    plt.plot(range(0, len(data_graph[graph_pages[i][0]])), data_graph[graph_pages[i][0]], marker='.', color=colors[i].hex_l, label=graph_pages[i][0], linestyle=linestyles[i%len(linestyles)])

lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Pages Selected")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.savefig(str(year_selected)+"_features_poisson.png", bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)

# Write information to file
with open(str(year_selected)+"_information_poisson.csv", 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['mse', mean_squared_error(labels_test, result_lcv)])
    for p in important_pages:
        spamwriter.writerow([p[0], float(p[1])])