#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LASSO model which uses wikipedia data to predicts ILI levels
Usage:
  generate_weekly_data.py <year>

Options:
  -h, --help
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from docopt import docopt
from colour import Color
import csv

from models_utils import generate, generate_labels, generate_labels_one_year, generate_one_year, cross_validation_glm, standardize_data, generate_keywords, stz

from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict
from cvglmnet import cvglmnet

#### INITIALIZATION #####

# Parse the command line
arguments = docopt(__doc__)

# Year selected
year_selected = int(arguments["<year>"])

# Selected columns that will be extracted from the dataframes
selected_columns = generate_keywords()

##### ALGORITHM ######

# Data generation from data files
dataset = generate(year_selected)[selected_columns]
labels = generate_labels(year_selected)["Incidenza Totale"]
data = generate_one_year(year_selected)[selected_columns]
labels_test = generate_labels_one_year(year_selected)["Incidenza Totale"]
weeks = generate_labels_one_year(year_selected)["Settimana"]

print ("------------")
print(year_selected)
print ("Train set shape (rows/columns): ", dataset.shape)
print ("Test set shape (rows/columns): ", data.shape)

# Create a copy of the dataset where we fills NaN values
# with 0.
dataset_zero = dataset.fillna(0);
data_zero = data.fillna(0);

# Create the poisson model (use cross-validation to obtain the best alpha value
model = cvglmnet(x=dataset_zero.as_matrix().copy(), y=labels.copy().as_matrix(), family='poisson', alpha=1.0, ptype="mse", parallel=True, nfolds=10)

# Predict
model_result = cvglmnetPredict(model, data_zero.as_matrix(), ptype ='response', s ="lambda_min")

# Get the features weights
coeff = cvglmnetCoef(model, s ="lambda_min")

# Regenerate panda dataframe with standardized data
data_graph = pd.DataFrame(data=data_zero, columns=selected_columns)

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
print ("Poisson MSE: ", mean_squared_error(labels_test, model_result))
print ("Poisson Standardized MSE: ", mean_squared_error(labels_test, model_result))
index_picco = labels_test.idxmax(axis=1)
index_picco_m = np.argmax(model_result)
index_picco_m = 11
print("Picco Influenzale:", weeks[index_picco])
print("Picco Influenzale Modello: ", weeks[index_picco_m])
print("Valore Picco: ", labels_test[index_picco])
print("Valore Picco Modello: ", model_result[index_picco_m])
print ("------------")
print()

# Plot some informations
plt.figure(figsize=(20, 10))
plt.ylabel("Incidenza su 1000 persone")
plt.xlabel("Settimane")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

plt.plot(range(0, len(model_result)), model_result, 'o-', label="Poisson Model")
plt.plot(range(0, len(labels_test)), labels_test, 'x-', label="Actual Value")

plt.legend()
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.savefig(str(year_selected)+"_poisson.png")

plt.figure(2, figsize=(20, 10))
plt.title("Andamento voci di Wikipedia rispetto ad Incidenza ILI", fontsize=18)
plt.ylabel("Incidenza standardizzata [0, 1]", fontsize=17)
plt.xlabel("Settimana", fontsize=17)
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical", fontsize=15)
plt.yticks(fontsize=15)

# All possible colors
num_feat=5
#linestyles = ['-', '--', '-.', ':']
colors =list(Color("blue").range_to(Color("red"), num_feat))
for i in range(0, num_feat):
    if i>=len(graph_pages):
        break;
    plt.plot(range(0, len(data[graph_pages[i][0]])), stz(data_zero[graph_pages[i][0]]), marker='.', color=colors[i].hex_l, label=graph_pages[i][0])#, linestyle=linestyles[i%len(linestyles)])
plt.plot(range(0, len(labels_test)), stz(labels_test), 'k-', label="Incidenza ILI", linewidth=5)
plt.grid()

lgd = plt.legend(fontsize=14)
plt.savefig(str(year_selected)+"_features_poisson.png", bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)

# Write information to file
with open(str(year_selected)+"_information_poisson.csv", 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['mse', mean_squared_error(labels_test, model_result)])
    for p in important_pages:
        spamwriter.writerow([p[0], float(p[1])])
