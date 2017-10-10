#!/bin/bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

#input_dir="/mnt/fluiddata/cconsonni/pagecounts/data/output"
input_dir="/mnt/fluiddata/cconsonni/pagecounts-new/pagecounts/data/output"

for year in {2007..2016}
do
    #if [ $year == 2007 ]; then
    # all_files="$input_dir/$year-*"
    #else
    # all_files="$input_dir/$year-*/output-*"
    #fi
    #./parse_dumps.sh -k "./../data/keywords/keywords_random.txt" -i "$all_files" -o "./output_random/$year" --debug
    ./generate_weekly_data.py ./output_random/$year/result.output ./output_random/$year/$year.csv -c ../data/keywords/keywords_random.txt
done
