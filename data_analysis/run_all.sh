#!/bin/bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

c=("italy" "netherlands" "germany" "austria" "belgium")
type_of_exec=("no-future" "future")
type_of_data=("old_data" "new_data" "pageviews")

for country in ${c[@]};
do
    mkdir -p ~/Desktop/summarized_results/$country

    echo "file_name, mse, r2_score, pcc" >> ~/Desktop/summarized_results/$country/comparative_analysis_$country.csv

	for file in `ls ~/Desktop/complete_results/*/*/*/*.csv | grep $country | grep prediction | sort`; 
	do
		./analyze_results.py $file >> ~/Desktop/summarized_results/$country/comparative_analysis_$country.csv
	done
done

for country in ${c[@]};
do
    for exec in ${type_of_exec[@]};
    do
        for data in ${type_of_data[@]};
        do
            if [ $exec == "future" ];
            then
                file=`ls ~/Desktop/complete_results/*/*/*/*.png | grep $country | grep $data | grep -w -v "no-future" | grep -v features | grep season`
            else
                file=`ls ~/Desktop/complete_results/*/*/*/*.png | grep $country | grep $data | grep -w "no-future" | grep -v features | grep season`
            fi

            command="cp $file ~/Desktop/summarized_results/$country/$exec\_$data\_$country.png"
            eval $command

        done
    done
done
