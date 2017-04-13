#!/bin/bash

input=${1:-}

function print_week {
    page=`echo ${1:-} | sed -e 's\,\ \g' |awk '{print $1}'`
    date_hour=`echo ${1:-} | sed -e 's\,\ \g' | awk '{print $2}'`
    date_only=`echo $date_hour | sed -e 's\-\ \g' | awk '{print $1}'`
    visits=`echo ${1:-} | sed -e 's\,\ \g' |awk '{print $3}'`
    week=`/bin/date -d "$date_only" +%V`

    echo "$page $week $visits"
}

export -f print_week
cat $input | parallel --no-notice print_week
