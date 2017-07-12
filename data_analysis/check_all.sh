#!/bin/bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

#input_dir="/mnt/fluiddata/cconsonni/pagecounts/data/output"
input_dir="/mnt/fluiddata/cconsonni/pagecounts-new/pagecounts/data/output"

for year in {2009..2016}
do
    all_files="$input_dir/$year-*/output-*"
    ./parse_dumps.sh -k "./../data/keywords3.txt" -i "$all_files" -o "./output3/$year" --debug
    ./generate_weekly_data.py ./output3/$year/result.output ./output3/$year/$year.csv -c ../data/keywords3.txt
done
