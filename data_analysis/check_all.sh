#!/bin/bash
# The purpose of this script is to generate the CSV files
# needed to train and validate the model. This simple
# bash lines runs for all the selected years and for each
# of then it parses the Wikipedia log and it generate
# the weekly data.
#
# Written by Giovanni De Toni (2017)
# Email: giovanni.det at gmail.com

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

# Directory where all the Wikipedia log are located
input_dir="/mnt/fluiddata/cconsonni/pagecounts-new/pagecounts/data/output"

# Directory where the generated files will be saved
output_dir="./output/"

# The keyword file which will be used to filter the Wikipedia logs
keywords_file="./../data/keywords.txt"

for year in {2009..2016}
do
    if [ $year == 2007]; then
       all_files="$input_dir/$year-*"
    else
        all_files="$input_dir/$year-*/output-*"
    fi
    ./parse_dumps.sh -k "$keywords_file" -i "$all_files" -o "$output_dir$year" --debug
    ./generate_weekly_data.py ".$output_dir/$year/result.output" "./output3/$year/$year.csv" -c "$keywords_file"
done
