#!/usr/bin/env bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

country_list=("germany" "austria" "italy" "netherlands")
year_lists="./year_lists"
type_of_data="new_data"
base_dir="./files"

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
        commanda="./model.py $start_year $end_year ./../data/wikipedia_${c}/$type_of_data ./../data/$c/$type_of_data ./../data/keywords/keywords_${c}.txt $c --f --d $directory --no-images"
        commandb="./model.py $((start_year+1)) $end_year ./../data/wikipedia_${c}/$type_of_data ./../data/$c/$type_of_data ./../data/keywords/keywords_${c}.txt $c --f --d $directory --no-images --no-future"
        echo $commanda
        eval $commanda
        echo $commandb
        eval $commandb
    done
done
