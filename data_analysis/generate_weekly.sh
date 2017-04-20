#!/bin/bash
# Given the result of parse_dumps scripts, this code
# subsitute each entry's date with the week number (from 1 to 54).
#
# Written by Giovanni De Toni (2017)
# Email: giovanni.det at gmail.com

# General variables
export input=''

# Usage and version information
eval "$(docopts -V - -h - : "$@" <<EOF
Usage: generate_weekly <input>

Options:
	<input>		Specify input file which will be parsed.
	--help			Show help options.
	--version		Print program version.
----
generate_weekly 0.1.0
Copyright (C) 2017 Giovanni De Toni
License MIT
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
EOF
)"

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

# This methods convert a date into its week number representation
function print_week {

	# Get the page name
    page=`echo ${1:-} | sed -e 's\,\ \g' |awk '{print $1}'`

	# Get the date hour and extract only year-month-day
	date_hour=`echo ${1:-} | sed -e 's\,\ \g' | awk '{print $2}'`
    date_only=`echo $date_hour | sed -e 's\-\ \g' | awk '{print $1}'`

	# Get total number of visit for the specific page
	visits=`echo ${1:-} | sed -e 's\,\ \g' |awk '{print $3}'`

	# Get the week number
	week=`/bin/date -d "$date_only" +%V`

	# Print everything
    echo "$page $week $visits"
}

# Export the function to make it visible from parallel
export -f print_week

# Convert the file
cat $input | parallel --no-notice print_week
