#!/usr/bin/env bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

country_list=("germany" "austria" "italy" "netherlands")
year_list=("2008-2016" "2013-2016" "2008-2016" "2010-2016")
year_list_future=("2008-2018" "2013-2018" "2008-2018" "2010-2018")
type_of_exec=("future" "no-future")
type_of_data=("old_data" "new_data" "pageviews")
base_dir="./complete_results"

for data_type in ${type_of_data[@]}
do
    for type_exec in ${type_of_exec[@]}
    do
        counter=0;
	    for c in ${country_list[@]}
	    do

            # Create the directory, if it doesn't exists
	        directory=$base_dir/$data_type/$type_exec/$c
    	    if [ ! -d $base_dir/$c ]; then
        	    mkdir -p $base_dir/$data_type/$type_exec/$c
    	    fi

            if [ $data_type == "old_data" ]; then
    	        start_year="$(echo ${year_list[$counter]} | cut -f 1 -d -)"
                end_year="$(echo ${year_list[$counter]} | cut -f 2 -d -)"
            elif [ $data_type == "new_data" ]; then
                start_year="$(echo ${year_list_future[$counter]} | cut -f 1 -d -)"
                end_year="$(echo ${year_list_future[$counter]} | cut -f 2 -d -)"
            else
                start_year="2016"
                end_year="2018"
            fi


            # Check which type of execution we need to perform.
            if [ $type_exec == "no-future" ]; then
                command="./model.py $start_year $end_year ./../data/wikipedia_${c}/$data_type ./../data/$c/$data_type ./../data/keywords/keywords_${c}.txt $c --f --d $directory --no-future"
            else
                command="./model.py $start_year $end_year ./../data/wikipedia_${c}/$data_type ./../data/$c/$data_type ./../data/keywords/keywords_${c}.txt $c --f --d $directory"
            fi

            # Evaluate the command
            echo $command
            eval $command

            counter=$counter+1
	    done

    done
done
