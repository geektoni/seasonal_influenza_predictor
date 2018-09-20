import glob

from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import models_utils

from glmnet import glmnet
from glmnetPredict import glmnetPredict
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict
from cvglmnet import cvglmnet
from glmnetPrint import glmnetPrint
from pyglmnet import GLM, GLMCV

import matplotlib.pyplot as plt

m1 = ['Week','1918_flu_pandemic', 'Avian_influenza',
       'Centers_for_Disease_Control_and_Prevention', 'Common_cold', 'Epidemic',
        'Fever',
       'Flu_season', 'Human_influenza', 'Influenza-like_illness',
       'Influenza_A_virus',
       'Influenza_A_virus_subtype_H2N2', 'Influenza_A_virus_subtype_H3N1',
       'Influenza_A_virus_subtype_H3N2', 'Influenza_A_virus_subtype_H5N1',
       'Influenza_A_virus_subtype_H5N2', 'Influenza_A_virus_subtype_H9N2',
       'Influenza_research', 'Influenza_treatment', 'Influenza_vaccine',
       'Influenza_virus', 'Influenzavirus_B', 'Influenzavirus_C',
       'Oseltamivir', 'Tamiflu']

week_to_remove = ["2009-17", "2009-18", "2009-19", "2009-20", "2009-17", "2012-40",
				  "2012-41","2012-42","2012-43","2012-44","2012-45","2012-46","2012-47","2012-48","2012-49",
				  "2012-50","2012-51","2012-52","2013-01","2013-02","2013-03","2013-04","2013-05","2013-06",
				  "2013-07","2013-08","2013-09","2013-10","2013-11", "2013-12", "2013-13"]

# Read incidence value
data="../data/USA/old_data/data.csv"
incidence = pd.read_csv(data)

# Read wikipedia value
wikipedia="../data/wikipedia_USA/old_data/*.csv"
dataset_wiki = pd.DataFrame()
file_list = sorted(glob.glob(wikipedia))
for file in file_list:
	if dataset_wiki.empty:
		dataset_wiki = pd.read_csv(file)
	else:
		tmp = pd.read_csv(file)
		dataset_wiki = pd.concat([dataset_wiki, tmp], ignore_index=True)

# Use the m1 model instead
#dataset_wiki = dataset_wiki[m1]

# Compose the final dataset which will be used for training
dataset_weeks = incidence.join(dataset_wiki)

# Drop high-volume seasons
dataset_weeks = dataset_weeks[~dataset_weeks.Week.isin(week_to_remove)]

# Set a new index to the wikipedia dataset and
# drop the week column
dataset = dataset_weeks.drop(["Week"], axis=1)

# Fill NaN values with zero
dataset_zero = dataset.fillna(0)

# Remove 2009 year
#dataset_zero = dataset_zero[dataset_zero.year != 2009]

# Convert the week number with the month
dataset_zero["full_date"] = pd.to_datetime(dataset_zero.year.astype(str), format='%Y') + \
             pd.to_timedelta(dataset_zero.week.mul(7).astype(str) + ' days')
dataset_zero["month"]= pd.DatetimeIndex(dataset_zero['full_date']).month
dataset_zero = dataset_zero.drop(["full_date", "week"], axis=1)

# List of the final values
total_results= []
total_incidence = []

for year in [2007, 2008, 2010, 2011, 2012, 2013]:

	train = dataset_zero[dataset_zero.year != year]
	test = dataset_zero[dataset_zero.year == year]

	#model = cvglmnet(x=train.drop(["incidence"], axis=1).values.copy(), y=train["incidence"].fillna(0).copy().values,
	#				 family='poisson', alpha=1.0, ptype='mse', parallel=True, nfolds=20)
	#result = cvglmnetPredict(model, test.drop(["incidence"], axis=1).values, ptype="response", s="lambda_min")

	model=glmnet(x=train.drop(["incidence"], axis=1).values.copy(), y=train["incidence"].fillna(0).copy().values,
					 family='poisson', lambdau=np.array([0, 1]), alpha=0)
	result = glmnetPredict(model, test.drop(["incidence"], axis=1).values)
	#glmnetPrint(model)
	result = [x[1] for x in result]

	total_results = total_results + result
	total_incidence = total_incidence + test["incidence"].tolist()

mse = mean_squared_error(total_results, total_incidence)
print("MSE: ", mse)

# Print R squared
r2 = r2_score(total_results, total_incidence)
print("R^2: ", r2)

# Print Pearson Coefficient
pcc = np.corrcoef(total_results, total_incidence)[0][1]
print("Pearson Correlation Coeff: ", pcc)

# Print the graph
plt.style.use('seaborn-whitegrid')
plt.figure(1, figsize=(15,6))
plt.plot(range(0, len(total_incidence)),total_incidence, color="black", label="Real Incidence")
plt.plot(range(0, len(total_results)), total_results, color="red", label="Predicteted Incidence")

plt.xticks(np.arange(0, len(dataset_weeks[dataset_weeks.year != 2009]["Week"]), 7), dataset_weeks[dataset_weeks.year != 2009]["Week"].tolist()[0::7], rotation="vertical")
plt.xlabel("Year-Week", fontsize=16)
plt.ylabel("%ILI", fontsize=16)
plt.legend(facecolor="white", edgecolor="black", fancybox=True)
plt.tight_layout()

plt.show()
