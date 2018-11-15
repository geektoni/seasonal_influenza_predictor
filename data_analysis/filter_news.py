"""Generate year files with news counts
Usage:
  filter_news.py <directory> <output>

Options:
  -h, --help
"""

from docopt import docopt

from os import listdir
from os.path import isfile, join

import datetime

from tqdm import *

import pandas as pd

def find_index(id, lis):
	for i in range(0, len(lis)):
		if id == lis[i]:
			return i
	return -1

if __name__ == "__main__":

	# Parse the command line
	args = docopt(__doc__)

	# Array with the week we are considering
	weeks = [42,43,44,45,46,47,48,49,50,51,52,1,23,4,5,6,7,8,9,10,11,12,13,14,15]

	# Final count dictionary
	news_count = {}

	# Get only the files in the directory
	onlyfiles = [f for f in listdir(args["<directory>"]) if isfile(join(args["<directory>"], f))]

	# Loop over all the files and parse them
	for file in tqdm(onlyfiles):

		# Split the filename and get the day/month/year
		file_name = file.split("_")
		day = file_name[2]
		month = file_name[1]
		year = file_name[3]

		# Compute the week number
		week_number = datetime.date(int(year), int(month), int(day)).isocalendar()[1]

		# Read and parse the file only if it is in the week range we are considering
		if week_number in weeks:

			# Read the file
			file = pd.read_csv(join(args["<directory>"], file))

			# Count how many news we have, considering only the italian ones
			total_news = file[file.lang_detected == "it"].count()

			# If that year column is still empty, create it and set it to zero
			if news_count.get(year, []) == []:
				news_count[year] = [0 for x in range(0, len(weeks))]

			# Increment the week count
			news_count[year][find_index(week_number, weeks)] += int(total_news["lang_detected"])

	# Generate the index for the future dataframe
	df_index = []

	# Add a zero in front of number less than 10
	for i in weeks:
		if i < 10:
			number = "0" + str(i)
		else:
			number = str(i)
		df_index.append(number)

	# Generate the dataframe
	final_df = pd.DataFrame(news_count)
	final_df.set_index(df_index)

	# Print the dataframe to show the result
	print(final_df)

	# Save it to file
	final_df.to_csv(args["<output>"], index_label="Week")
