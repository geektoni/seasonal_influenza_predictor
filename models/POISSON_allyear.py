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

from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict
from cvglmnet import cvglmnet

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

range_sel = (2014, 2017)

for year_selected in range(range_sel[0], range_sel[1]):

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
    fit = cvglmnet(x=train.as_matrix().copy(), y=labels.copy().as_matrix(), family='poisson', alpha=1.0,
                   ptype="mse", parallel=True, nfolds=10)
    result_lcv = cvglmnetPredict(fit, test.as_matrix(), ptype='response', s="lambda_min")
    coeff = cvglmnetCoef(fit, s="lambda_min")

    all_predicted_values.extend(result_lcv)
    all_true_labels.extend(labels_test)
    total_weeks.extend(weeks)

# Prim MSE of the two models
print("------------")
print("LASSO XVal  General MSE: ", mean_squared_error(all_true_labels, all_predicted_values))
print("------------")

# Plot some informations
fig = plt.figure(3, figsize=(15, 6))
ax = fig.add_subplot(111)

plt.title("Stagioni influenzali Ottobre "+str(range_sel[0])+" - Aprile "+str(range_sel[1]), fontsize=18)
plt.ylabel("Incidenza su 1000 persone", fontsize=17)
plt.xlabel("Anno-Settimana", fontsize=17)

plt.xticks(range(0, len(total_weeks)), total_weeks, rotation="vertical", fontsize=15)
plt.yticks(fontsize=15)

plt.plot(range(0, len(all_predicted_values)), all_predicted_values, 'r-', label="Modello Lineare", linewidth=3)
plt.plot(range(0, len(all_true_labels)), all_true_labels, 'k-', label="InfluNet", linewidth=3)

plt.grid()
plt.legend(fontsize=14)
plt.tight_layout()

plt.savefig("appendix_poisson_"+str(range_sel[0])+"-"+str(range_sel[1])+".png", dpi=150)