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

from models_utils import generate, generate_labels, generate_labels_one_year, generate_one_year, standardize_data, generate_keywords

#### INITIALIZATION #####

# Parse the command line
#arguments = docopt(__doc__)

# Year selected
#year_selected = int(arguments["<year>"])

# Selected columns that will be extracted from the dataframes
selected_columns = generate_keywords()

##### ALGORITHM ######

all_predicted_values = []
all_true_labels = []
total_weeks = []

for year_selected in range(2014, 2017):

    print("[*] ", year_selected )
    # Data generation from data files
    dataset = generate(year_selected)[selected_columns]
    labels = generate_labels(year_selected)["Incidenza Totale"]
    data = generate_one_year(year_selected)[selected_columns]
    labels_test = generate_labels_one_year(year_selected)["Incidenza Totale"]
    weeks = generate_labels_one_year(year_selected)["Settimana"]

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

    all_predicted_values.extend(result_lcv_imp)
    all_true_labels.extend(labels_test)
    total_weeks.extend(weeks)

# Prim MSE of the two models
print("------------")
print("LASSO XVal  General MSE: ", mean_squared_error(all_true_labels, all_predicted_values))
print("------------")

# Plot some informations
fig = plt.figure(3, figsize=(15, 6))
ax = fig.add_subplot(111)

plt.title("Stagioni influenzali Ottobre 2013 - Aprile 2016", fontsize=18)
plt.ylabel("Incidenza su 1000 persone", fontsize=17)
plt.xlabel("Anno-Settimana", fontsize=17)

weeks_used = []
for k, v in enumerate(total_weeks):
    if k%3==0:
        weeks_used.append(v)

plt.xticks(range(0, len(total_weeks)), total_weeks, rotation="vertical", fontsize=15)
plt.yticks(fontsize=15)
#start, end = ax.get_xlim()
#ax.xaxis.set_ticks(np.arange(start, end, 3))
#plt.xticks(weeks_used, rotation="vertical", fontsize=15)


plt.plot(range(0, len(all_predicted_values)), all_predicted_values, 'r-', label="Modello Lineare", linewidth=3)
plt.plot(range(0, len(all_true_labels)), all_true_labels, 'k-', label="InfluNet", linewidth=3)

plt.grid()
plt.legend(fontsize=14)
plt.tight_layout()

plt.savefig("chapter_3_linear_2013-2016.png", dpi=150)

# Write information to file
#with open(str(year_selected)+"_information.csv", 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    spamwriter.writerow(['mse', mean_squared_error(labels_test, result_lcv_imp)])
#    spamwriter.writerow(['alpha', mean_squared_error(labels_test, result_lcv_imp)])
#    for p in important_pages:
#        spamwriter.writerow([p[0], p[1]])