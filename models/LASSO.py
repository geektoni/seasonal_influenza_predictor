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
import matplotlib
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from colour import Color
from docopt import docopt
import csv

from models_utils import generate, generate_labels, generate_labels_one_year, generate_one_year, standardize_data, generate_keywords

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

print("Train set shape (rows/columns): ", dataset.shape)
print("Test set shape (rows/columns): ", data.shape)
print("------------")

# Create a copy of the dataset where we fills NaN values
# with 0.
dataset_zero = dataset.fillna(0);
data_zero = data.fillna(0);

# Standardize data
train, test = standardize_data(dataset_zero, data_zero)

# Regenerate panda dataframe with standardized data
data_graph = pd.DataFrame(data=test, columns=selected_columns);

# Create a Lasso Cross-Validation instance which will be
# trained on the dataset in which NaN values are replaced
# with the column mean.
lassoCV_imp = LassoCV(max_iter=100000, n_jobs=-1, n_alphas=1000)
lassoCV_imp.fit(train, labels)
result_lcv_imp = lassoCV_imp.predict(test)
print("LassoCV Standardized alpha: ", lassoCV_imp.alpha_)

#model_lasso = Lasso(alpha=1.0, max_iter=100000)
#model_lasso.fit(dataset_zero, labels)
#result_lasso = model_lasso.predict(data_zero)

# Extract which features seems to be important
# from the LASSO model
important_pages=[]
graph_pages=[]
for i in list(zip(lassoCV_imp.coef_, selected_columns)):
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
print("------------")
print("Pages which their weight is != from 0: ")
print(tabulate(important_pages, headers=["Page", "Mean"]))

# Prim MSE of the two models
print("------------")
print("LASSO XVal MSE: ", mean_squared_error(labels_test, result_lcv_imp))
print("------------")

# Plot some informations
font = {'size': 20}

matplotlib.rc('font', **font)

plt.figure(1, figsize=(20, 10))
plt.title("Stagione influenzale "+str(year_selected-1)+"-"+str(year_selected))
plt.ylabel("Incidenza su 1000 persone")
plt.xlabel("Settimane")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

plt.plot(range(0, len(result_lcv_imp)), result_lcv_imp, 'd-', label="Lasso Model (alpha selected by X-val)")
plt.plot(range(0, len(labels_test)), labels_test, 'x-', label="Actual Value")

lgd = plt.legend(loc=1, borderaxespad=0.)
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.savefig(str(year_selected)+".png", bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)

plt.figure(2, figsize=(20, 10))
plt.ylabel("Numero di visite")
plt.xlabel("Settimane")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

# All possible colors
linestyles = ['-', '--', '-.', ':']
colors =list(Color("blue").range_to(Color("red"), 10))
for i in range(0, 10):
    if i>=len(graph_pages):
        break;
    plt.plot(range(0, len(data[graph_pages[i][0]])), data[graph_pages[i][0]], marker='.', color=colors[i].hex_l, label=graph_pages[i][0], linestyle=linestyles[i%len(linestyles)])

lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Pagine selezionate")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.savefig(str(year_selected)+"_feature.png", bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)

# Write information to file
with open(str(year_selected)+"_information.csv", 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['mse', mean_squared_error(labels_test, result_lcv_imp)])
    spamwriter.writerow(['alpha', mean_squared_error(labels_test, result_lcv_imp)])
    for p in important_pages:
        spamwriter.writerow([p[0], p[1]])