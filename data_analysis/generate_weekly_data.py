#!/usr/bin/env python
# Given a complete year files with data in the form (page, week, visits)
# this script will generate a convenient csv file which will store for
# each page and for each years's week the total number of visits.
#
# Written by Giovanni De Toni (2017)
# Email: giovanni.det at gmail.com

"""Generate year files with page visits
Usage:
  generate_weekly_data.py <input> <output> [-c <columns>]

Options:
  -c    specify custom columns name.
  -h, --help
"""

import fileinput
import requests
import json
import pandas as pd
import numpy as np
import datetime
from tqdm import *
from docopt import docopt

# Parse the command line
arguments = docopt(__doc__)

# Set up an empty dictionary
all_data={}

# Set up the wikipedia API link to check when a page was created
wiki_url = "https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvlimit=1&rvprop=timestamp&rvdir=newer&format=json&titles="

# If the columns are set, add them to the dataframe
columns_df = []
with open(arguments["<columns>"], "r") as f:
    for line in f:
        c_name = line.replace("\n", "").replace("\\", "")
        columns_df.append(c_name)
        all_data[c_name]=[0 for x in range(53)]

# Future index
index_year=[];

# Read from file
with open(arguments["<input>"], "r") as f:
    for line in f:

        # Split the line given
        # 0: page name
        # 1: date-hour
        # 2: visits count
        # 3: page size
        total = line.split(",")

        # Create the page only with specified columns
        found = False
        for p in columns_df:
            if (p == total[0].replace("\\", "")):
                found = True
                break

        if (found):

            # Get date-hour pair
            # 0: date
            # 1: hour
            date = total[1].split("-")

            # Generate year month and day.
            year = date[0][0:4]
            month = date[0][4:6]
            day = date[0][6:8]

            # Get week number
            week_number = datetime.date(int(year), int(month), int(day)).isocalendar()[1]

            # Set up an empty list if the key
            # is null
            if all_data.get(total[0].replace("\\", ""), []) == []:
                all_data[total[0].replace("\\", "")] = [0 for x in range(53)]

            # Sum the visits
            try:
                all_data[total[0].replace("\\", "")][int(week_number)-1] += int(total[2]);
            except ValueError:
                print("[*] Catch value error here: {}".format(total[2]))


for i in range(1, 54):
	if i<10:
		number="0"+str(i)
	else:
		number=str(i)
	index_year.append(year+"-"+number)

# Generate a pandas dataframe with all the data
print(index_year)
df = pd.DataFrame(all_data);
df = df.set_index([index_year]);

# Go through all data and set column value to NaN if the page was created
# year after
for c in tqdm(df.columns):
    df[c] = df[c].astype(np.float64)
    r = requests.get(wiki_url+str(c))
    wiki_data = json.loads(r.text)
    for i in range(0, 53):
        # Generate week number original
        week_number = df.index[i]
        year_orig = week_number.split("-")[0]
        week_orig = week_number.split("-")[1]
        for key, value in wiki_data["query"]["pages"].items():
            if key != u'-1':
                week_str = value["revisions"][0]["timestamp"]

                # Generate year month and day.
                date = week_str.split("T")[0].split("-");
                year = date[0]
                month = date[1]
                day = date[2]
                week = datetime.date(int(year), int(month), int(day)).isocalendar()[1]

                if (int(year) > int(year_orig) or int(week) > int(week_orig)):
                    df.loc[week_number, c] = np.NaN

# Print the dataframe to show the result
print(df)

# Save it to file
df.to_csv(arguments["<output>"], index_label="Week")
