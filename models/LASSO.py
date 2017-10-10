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
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from colour import Color
from docopt import docopt
import csv

from models_utils import generate, generate_labels, generate_labels_one_year, generate_one_year, standardize_data, generate_keywords, stz

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

print("------------")
print(year_selected)
print("Train set shape (rows/columns): ", dataset.shape)
print("Labels (row/columns): ", labels_test.shape)
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
model = LassoCV(max_iter=100000, n_jobs=-1, n_alphas=1000)
model.fit(train, labels)
model_results = model.predict(test)

# Extract which features seems to be important
# from the LASSO model
important_pages=[]
graph_pages=[]
for i in list(zip(model.coef_, selected_columns)):
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
print("Pages which their weight is != from 0: ")
print(tabulate(important_pages, headers=["Page", "Mean"]))

# Prim MSE of the two models
print("------------")
print("LASSO XVal Alpha: ", model.alpha_)
print("LASSO XVal MSE: ", mean_squared_error(labels_test, model_results))
index_picco = labels_test.idxmax(axis=1)
index_picco_m = np.argmax(model_results)
print("Picco Influenzale:", weeks[index_picco])
print("Picco Influenzale Modello: ", weeks[index_picco_m])
print("Valore Picco: ", labels_test[index_picco])
print("Valore Piccoo Modello: ", model_results[index_picco_m])
print("------------")
print()

# Plot some informations
font = {'size': 18}

matplotlib.rc('font', **font)

plt.figure(1, figsize=(15, 10))
plt.title("Stagione influenzale Ottobre "+str(year_selected-1)+" - Aprile "+str(year_selected))
plt.ylabel("Incidenza su 1000 persone")
plt.xlabel("Settimane")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

plt.plot(range(0, len(model_results)), model_results, 'd-', label="Lasso Model (alpha selected by X-val)")
plt.plot(range(0, len(labels_test)), labels_test, 'x-', label="Incidenza ILI")

lgd = plt.legend(loc=1, borderaxespad=0.)
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.savefig(str(year_selected)+".png", bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)

fig = plt.figure(2, figsize=(30, 10))
ax = fig.add_subplot(111)
plt.title("Andamento feature rispetto ad incidenza ILI")
plt.ylabel("Incidenza standardizzata")
plt.xlabel("Settimana")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

# All possible colors
num_feat=5
linestyles = ['-', '--', '-.', ':']
colors =list(Color("blue").range_to(Color("red"), num_feat))
for i in range(0, num_feat):
    if i>=len(graph_pages):
        break;
    plt.plot(range(0, len(data[graph_pages[i][0]])), stz(data_zero[graph_pages[i][0]]), marker='.', color=colors[i].hex_l, label=graph_pages[i][0])#, linestyle=linestyles[i%len(linestyles)])
plt.plot(range(0, len(labels_test)), stz(labels_test), 'k-', label="Incidenza ILI", linewidth=5)
plt.grid()

lgd = plt.legend()
plt.savefig(str(year_selected)+"_feature.png", bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=150)

# Write information to file
with open(str(year_selected)+"_information.csv", 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['mse', mean_squared_error(labels_test, model_results)])
    spamwriter.writerow(['alpha', model.alpha_])
    for p in important_pages:
        spamwriter.writerow([p[0], p[1]])