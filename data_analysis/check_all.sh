#!/bin/bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

input_dir="/mnt/fluiddata/cconsonni/pagecounts/data/output"

for year in {2009..2016}
do
    all_files="$input_dir/$year-*/output-*"
    ./parse_dumps.sh -k "./../data/keywords.txt" -i "$all_files" -o "./output/$year" --debug
    ./generate_weekly_data.py ./output/$year/result.output ./output/$year/$year.csv
done
