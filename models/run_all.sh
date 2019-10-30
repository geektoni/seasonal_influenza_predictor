#!/usr/bin/env bash

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

country_list=("belgium" "germany" "italy" "netherlands" "USA")
year_list=("2011-2016" "2008-2016" "2008-2016" "2010-2016" "2008-2013")
year_list_future=("2011-2019" "2008-2019" "2008-2019" "2010-2019" "2008-2013")
type_of_data=("cyclerank" "new_data" "cyclerank_pageviews" "pageviews" "old_data")
type_of_exec=("future" "no-future")
base_dir="./complete_results"

for data_type in ${type_of_data[@]}
do
    for type_exec in ${type_of_exec[@]}
    do
        counter=0;
	    for c in ${country_list[@]}
	    do

        if [ $c == "USA" ] && [ $type_exec != "old_data" ] && [ $type_exec != "pageviews" ]; then
          break
        fi
            # Create the directory, if it doesn't exists
	        directory=$base_dir/$data_type/$type_exec/$c/
    	    if [ ! -d $base_dir/$c ]; then
        	    mkdir -p $base_dir/$data_type/$type_exec/$c
    	    fi

            if [ $data_type == "old_data" ]; then
    	        start_year="$(echo ${year_list[$counter]} | cut -f 1 -d -)"
                end_year="$(echo ${year_list[$counter]} | cut -f 2 -d -)"
            elif [ $data_type == "new_data" ] || [ $data_type == "cyclerank" ]; then
                start_year="$(echo ${year_list_future[$counter]} | cut -f 1 -d -)"
                end_year="$(echo ${year_list_future[$counter]} | cut -f 2 -d -)"
			      else
                start_year="2016"
                end_year="2019"
            fi


            # Check which type of execution we need to perform.
            incidence_data_type=""
            keywords_ending=""
            if [ $data_type == "cyclerank" ]; then
              if [ $c == "USA" ]; then
                incidence_data_type="old_data"
              else
                incidence_data_type="new_data"
              fi
              keywords_ending="cyclerank_${c}"
			      elif [ $data_type == "cyclerank_pageviews" ]; then

				          incidence_data_type="pageviews"
                  keywords_ending="cyclerank_${c}"
            else
              	incidence_data_type=$data_type
              	keywords_ending=${c}
            fi

            if [ $type_exec == "no-future" ]; then
                start_year="$(expr $start_year + 1)"
                command="./model.py $start_year $end_year ./../data/wikipedia_${c}/$data_type ./../data/$c/$incidence_data_type ./../data/keywords/keywords_${keywords_ending}.txt $c --f --d $directory --no-future"
            else
                command="./model.py $start_year $end_year ./../data/wikipedia_${c}/$data_type ./../data/$c/$incidence_data_type ./../data/keywords/keywords_${keywords_ending}.txt $c --f --d $directory"
            fi

            # Evaluate the command
            echo $command
            eval $command

            counter=$counter+1
	    done

    done
done
