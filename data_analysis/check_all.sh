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

input_dir=''
output_dir=''
language=''
keywords_file=''
start_year=''
end_year=''

# Usage and version information
eval "$(docopts -V - -h - : "$@" <<EOF
Usage: check_all -i <input_dir> [-o <output_dir>] [-l <language>] -k <keywords_file> [-s <start_year>] [-e <end_year>]

Options:
	-i <input>		    Specify input directory.
	-o <output>		    Specify output directory (default is ./output).
	-l <language>	    	    The language of Wikipedia dumps we want to monitor (two letters).
	-k <keywords_file>          The keyword file.
	-s <start_year>             Start year of data (default is 2007).
	-e <end_year>               Last year of data (default is 2016).
	--help			    Show help options.
	--version		    Print program version.
----
check_all 0.1.0
Copyright (C) 2017 Giovanni De Toni
License MIT
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
EOF
)"

# Add default value if -o was not set
if [ -z $output_dir ]; then output_dir='./output'; fi
if [ -z $start_year ]; then start_year='2007'; fi
if [ -z $end_year ]; then end_year='2016'; fi
if [ -z $language ]; then language='it'; fi


# Directory where all the Wikipedia log are located
#input_dir="/mnt/fluiddata/cconsonni/pagecounts-new/pagecounts/data/output"
#input_dir="/mnt/fluiddata/cconsonni/pagecounts-new/pageviews/data/output/download/tmp/pagecounts/data/output"

# Language of Wikipedia pages (like it, nl, de, etc.)
#language="de"

# Directory where the generated files will be saved
#output_dir="./output"

# The keyword file which will be used to filter the Wikipedia logs
#keywords_file="./../data/keywords/keywords_germany.txt"

for year in $( seq $start_year $end_year )
do
    if [ $year == 2007 ]; then
       	all_files="$input_dir/$year-*"
       	#all_files="$input_dir/output-$year*"
    else
        #all_files="$input_dir/$year-*/output-*"
    	all_files="$input_dir/$year-*"
    fi
    ./parse_dumps_bare.sh -k "$keywords_file" -i "$all_files" -o "$output_dir/$year" -l "$language" --debug
    ./generate_weekly_data.py "$output_dir/$year/result.output" "$output_dir/$year/$year.csv" -c "$keywords_file"
done
