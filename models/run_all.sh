#!/usr/bin/env bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

country_list=("italy" "austria" "germany" "netherlands")
year_lists="./year_lists"
type_of_data="old_data"
base_dir="./images"

# Generate all directories
for c in ${country_list[@]}
do
    if [ ! -d $base_dir/$c ]; then
        mkdir -p $base_dir/$c
    fi
done


for c in ${country_list[@]}
do
    for line in `cat $year_lists/$c.txt`
    do
        start_year=$(cut -d'-' -f1 <<< $line)
        end_year=$(cut -d'-' -f2 <<< $line)
        directory=$base_dir/$c/
        command="./model.py $start_year $end_year ./../data/wikipedia_$c/$type_of_data ./../data/$c/$type_of_data ./../data/keywords/keywords_$c.txt --f --d $directory"
        echo $command
        eval $command
    done
done