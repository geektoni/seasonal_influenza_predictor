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
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
import os
from docopt import docopt

#### INITIALIZATION #####

# Parse the command line
arguments = docopt(__doc__)

# Year selected
year_selected = int(arguments["<year>"])

# Feature path and labels
path_features = "./../data/wikipedia";
path_labels = "./../data/influnet/csv";

# Selected columns that will be extracted from the dataframes
selected_columns = []
file_ = open("../data/keywords2.txt", "r")
for line in file_:
    if line != "Week":
        selected_columns.append(line.replace("\n", "").replace("\\", ""))
selected_columns = [x.decode('utf-8') for x in selected_columns]

##### UTILITIES #######

def generate_features(year_a, year_b):
    if not year_a.empty:
        first_part= year_a[selected_columns].copy()[42:52]
    else:
        first_part = pd.DataFrame()
    if not year_b.empty:
        second_part= year_b[selected_columns].copy()[0:15]
    else:
        second_part = pd.DataFrame()

    return first_part.append(second_part)

# Generate feature list from the files. It takes also
# a year which will be not included.
def generate(stop_year):
    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_features)
    file_list.sort()

    for i in range(0, len(file_list)-1):
        # If the file's year is equal than stop_year then do anything
        if int(file_list[i].replace(".csv", "")) != stop_year-1:
            tmp_a = pd.read_csv(os.path.join(path_features, file_list[i]), encoding = 'utf8', delimiter=',')
        else:
            tmp_a = pd.DataFrame()

        if int(file_list[i+1].replace(".csv", "")) != stop_year:
            tmp_b = pd.read_csv(os.path.join(path_features, file_list[i+1]), encoding = 'utf8', delimiter=',')
        else:
            tmp_b = pd.DataFrame()

        # If the dataset is empty the generate a new dataframe.
        # Append a new dataframe if the dataset is not empty.
        if dataset.empty:
            dataset = generate_features(tmp_a, tmp_b)
        else:
            dataset = dataset.append(generate_features(tmp_a, tmp_b))

    return dataset

def generate_one_year(year):
    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_features)
    file_list.sort()

    for i in range(0, len(file_list)-1):
        if int(file_list[i].replace(".csv", "")) == year-1:
            tmp_a = pd.read_csv(os.path.join(path_features, file_list[i]), encoding = 'utf8', delimiter=',')
        else:
            tmp_a = pd.DataFrame()

        if int(file_list[i+1].replace(".csv", "")) == year:
            tmp_b = pd.read_csv(os.path.join(path_features, file_list[i+1]), encoding = 'utf8', delimiter=',')
        else:
            tmp_b = pd.DataFrame()

        # If the dataset is empty the generate a new dataframe.
        # Append a new dataframe if the dataset is not empty.
        if dataset.empty:
            dataset = generate_features(tmp_a, tmp_b)
        else:
            dataset = dataset.append(generate_features(tmp_a, tmp_b))
    return dataset

def generate_labels(stop_year):
    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_labels)
    file_list.sort()

    for i in range(0, len(file_list)):
        if (file_list[i] != "tabula-2006_2007.csv"):
            # Read the file
            _file = pd.read_csv(os.path.join(path_labels, file_list[i]))

            # Append data without the stop year
            years = file_list[i].replace("tabula-", "").split("_")
            if int(years[1].replace(".csv", "")) != stop_year:
                dataset = dataset.append(_file[0:11])
            if int(years[0]) != stop_year-1:
                dataset = dataset.append(_file[11:25])

    return dataset

def generate_labels_one_year(stop_year):
    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_labels)
    file_list.sort()

    for i in range(0, len(file_list)):
        if (file_list[i] != "tabula-2006_2007.csv"):
            # Read the file
            _file = pd.read_csv(os.path.join(path_labels, file_list[i]))

            # Append data without the stop year
            years = file_list[i].replace("tabula-", "").split("_")
            if int(years[1].replace(".csv", "")) == stop_year:
                dataset = dataset.append(_file[0:11])
            if int(years[0]) == stop_year-1:
                dataset = dataset.append(_file[11:25])

    return dataset

##### ALGORITHM ######

# Data generation from data files
dataset = generate(year_selected)
labels = generate_labels(year_selected)["Incidenza Totale"]
data = generate_one_year(year_selected)
labels_test = generate_labels_one_year(year_selected)["Incidenza Totale"]
weeks = generate_labels_one_year(year_selected)["Settimana"]

print "Train set shape (rows/columns): ", dataset.shape
print "Test set shape (rows/columns): ", data.shape
print "------------"

# Imputer for missing values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(dataset)
dataset_imp = imp.transform(dataset)
imp.fit(data.fillna(0))
data_imp = imp.transform(data.fillna(0))

# Create a copy of the dataset where we fills NaN values
# with 0.
dataset_zero = dataset.fillna(0);
data_zero = data.fillna(0);

# Create a Lasso Cross-Validation instance which will be
# trained on the dataset in which NaN values are replaced
# with 0.
lassoCV = LassoCV(max_iter=100000, n_jobs=-1)
lassoCV.fit(dataset_zero, labels)
result_lcv = lassoCV.predict(data_zero)
print "LassoCV alpha: ", lassoCV.alpha_

# Create a Lasso Cross-Validation instance which will be
# trained on the dataset in which NaN values are replaced
# with the column mean.
lassoCV_imp = LassoCV(max_iter=100000, n_jobs=-1)
lassoCV_imp.fit(dataset_imp, labels)
result_lcv_imp = lassoCV_imp.predict(data_imp)
print "LassoCV Imp alpha: ", lassoCV_imp.alpha_

# Extract which features seems to be important
# from the LASSO model
important_pages=[]
for i in list(zip(lassoCV_imp.coef_, selected_columns)):
    if (i[0] != 0):
        important_pages.append([i[0], i[1]]);

# Print important pages
print "------------"
print "Pages which their weight is != from 0: "
for i in important_pages:
    print "* ", i[1], "Weight: ", i[0]

# Prim MSE of the two models
print "------------"
print "LASSO MSE: ", mean_squared_error(labels_test, result_lcv)
print "LASSO with imputed NaN: ", mean_squared_error(labels_test, result_lcv_imp)

# Plot some informations
plt.ylabel("Incidenza su 1000 persone")
plt.xlabel("Settimane")
plt.xticks(range(0, len(weeks)), weeks, rotation="vertical")

plt.plot(range(0, len(result_lcv)), result_lcv, 'o-', label="LassoCV Model")
plt.plot(range(0, len(result_lcv_imp)), result_lcv_imp, 'd-', label="LassoCV Model Normalized")
plt.plot(range(0, len(labels_test)), labels_test, 'x-', label="Actual Value")

plt.legend()
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.show()
