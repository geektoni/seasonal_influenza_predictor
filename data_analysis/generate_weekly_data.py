#!/usr/bin/env python
# Given a complete year files with data in the form (page, week, visits)
# this script will generate a convenient csv file which will store for
# each page and for each years's week the total number of visits.
#
# Written by Giovanni De Toni (2017)
# Email: giovanni.det at gmail.com

"""Generate year files with page visits
Usage:
  generate_weekly_data.py <input> <output>

Options:
  -h, --help
"""

import fileinput
import pandas as pd
import datetime
from docopt import docopt

# Parse the command line
arguments = docopt(__doc__)

# Set up an empty dictionary
all_data={}

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
        if all_data.get(total[0], []) == []:
            all_data[total[0]] = [0 for x in range(53)]

        # Sum the visits
        all_data[total[0]][int(week_number)-1] += int(total[2]);


for i in range(0, 53):
	if i<10:
		number="0"+str(i)
	else:
		number=str(i)
	index_year.append(year+"-"+number)

# Generate a pandas dataframe with all the data
print index_year
df = pd.DataFrame(all_data);
df = df.set_index([index_year]);

# Print the dataframe to show the result
print(df)

# Save it to file
df.to_csv(arguments["<output>"], index_label="Week")
