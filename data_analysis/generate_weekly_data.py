#!/usr/bin/env python
# Given a complete year files with data in the form (page, week, visits)
# this script will generate a convenient csv file which will store for
# each page and for each years's week the total number of visits.
#
# Written by Giovanni De Toni (2017)
# Email: giovanni.det at gmail.com

import fileinput
import pandas as pd

# Set up an empty dictionary
all_data={}

# Read from standard input
for line in fileinput.input():

    # Split the line given
    # 0: page name
    # 1: week number
    # 2: visits on that week
    result = str.split(line)

    # Set up an empty list if the key
    # is null
    if all_data.get(result[0], []) == []:
        all_data[result[0]] = [0 for x in range(53)]

    # Sum the visits
    all_data[result[0]][int(result[1])] += int(result[2]);

# Generate a pandas dataframe with all the data
df = pd.DataFrame(all_data);

# Print the dataframe to show the result
print(df)

# Save it to file
df.to_csv("result.csv", index_label="Week")
