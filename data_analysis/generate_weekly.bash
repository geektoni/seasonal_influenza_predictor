#!/bin/bash

input=${1:-}

for line in `cat $input`
do
    page=`echo $line | sed -e 's\,\ \g' |awk '{print $1}'`
    date_hour=`echo $line | sed -e 's\,\ \g' | awk '{print $2}'`
    date_only=`echo $date_hour | sed -e 's\-\ \g' | awk '{print $1}'`
    visits=`echo $line | sed -e 's\,\ \g' |awk '{print $3}'`
    week=`/bin/date -d "$date_only" +%V`

    echo "$page $week $visits"
done
