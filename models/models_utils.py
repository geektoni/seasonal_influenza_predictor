import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import glmnet_python
from cvglmnet import cvglmnet
from cvglmnetPredict import cvglmnetPredict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

##### UTILITIES #######

#  Feature path and labels
path_features = "./../data/wikipedia_470"
path_labels = "./../data/influnet/csv"
keywords = "../data/keywords/keywords2.txt"

def generate_keywords():
    selected_columns = []
    file_ = open(keywords, "r")
    for line in file_:
        if line != "Week":
            selected_columns.append(line.replace("\n", "").replace("\\", ""))
    return selected_columns

def generate_features(year_a, year_b):
    if not year_a.empty:
        first_part= year_a.copy()[41:52]
    else:
        first_part = pd.DataFrame()
    if not year_b.empty:
        second_part= year_b.copy()[0:15]
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
        if int(file_list[i].replace(".csv", "")) != stop_year-1 and int(file_list[i].replace(".csv", "")) != 2007:
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
        if int(file_list[i].replace(".csv", "")) == year-1 and int(file_list[i].replace(".csv", "")) != 2007:
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
            if int(years[1].replace(".csv", "")) != stop_year and int(years[0]) != 2007:
                dataset = dataset.append(_file[0:11])
            if int(years[0]) != stop_year-1:
                dataset = dataset.append(_file[11:26])
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
            if int(years[1].replace(".csv", "")) == stop_year and int(years[0]) != 2007:
                dataset = dataset.append(_file[0:11])
            if int(years[0]) == stop_year-1:
                dataset = dataset.append(_file[11:26])

    return dataset

def standardize_data(train, test):
    data_total = np.concatenate((train, test), axis=0)
    dmean = data_total.mean(axis=0)
    dmax = data_total.max(axis=0)
    dmin = data_total.min(axis=0)
    dmax_min = dmax - dmin
    dataset_imp = (train - dmean) / dmax_min
    data_imp = (test - dmean) / dmax_min
    dataset_imp[np.isnan(dataset_imp)] = 0
    data_imp[np.isnan(data_imp)] = 0
    return (dataset_imp, data_imp)

def sort_cv(item):
    return item[2]

def cross_validation_glm(dataset, labels, data_imp, labels_test, runs=100):
    values=[]
    alphas = np.linspace(0.0, 1.0, num=runs)
    for i in tqdm(range(runs)):
        model = cvglmnet(x=dataset.copy(), y=labels.copy().as_matrix(), family='poisson', alpha=alphas[i], ptype="mse", parallel=True, nfolds=10)
        result_model = cvglmnetPredict(model, data_imp, ptype = 'response', s = 'lambda_min')
        score = mean_squared_error(labels_test, result_model)
        values.append([alphas[i], model.copy(), score])
    values = sorted(values, key=sort_cv)
    return values