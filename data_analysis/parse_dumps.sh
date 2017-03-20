#!/bin/bash
# Parse Wikipedia dumps and get information of specific
# pages, based on custom generated regexps.
#
# Written by Giovanni De Toni (2017)

# Set up strict bash
# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

# Usage and version information
usage="
Usage: parse_dumps [-o <file>] [-i <file>] [--debug]
Options:
	-o <output>  		Specify output directory
	-i <input>   		Specify input directory
	-d	 				Run with debug option.
	-h					Show help options.
	-v		  			Print program version.
"

version=<<EOF
parse_dumps 0.1.0
Copyright (C) 2017 Giovanni De Toni
License MIT
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
EOF

# General variables
debug=0
input="./data"
output="./output"

# Parse command line arguments
while getopts ":i:o:dvh" opt; do
	case $opt in
		i)
			input=$OPTARG
      		;;
		o)
			output=$OPTARG
			;;
		d)
			debug=1
			;;
		v)
			#FIXME
			echo $version
			exit 0
			;;
		h)
			#FIXME
			echo $usage
			exit 0
			;;
    	\?)
      		echo "Invalid option: -$OPTARG" >&2
      		exit 1
      		;;
    	:)
      		echo "Option -$OPTARG requires an argument." >&2
      		exit 1
      		;;
  esac
done

# Set up utility variables
# and debug function
function print_debug {
	if [ $debug -ne 0 ]; then echo ${1:-}; fi
}

print_debug "[*] Debug value --> $debug"
print_debug "[*] Input directory value --> $input"
print_debug "[*] Otput directory value --> $output"

# All Wikipedia dump files
dumps_files=`ls $input`

# Set keywords, project and an other
# section to add other constraints.
project="it"
keywords=(
	"Influenza"
	"Virus"
	"Vaccino"
	"Febbre"
)
other="."

# Check if the output directory exists
if [ ! -d $output ]; then
	mkdir $output
fi

# Generate the regexp concatenating
# all the keyword
regexp="$project ("
for key in "${keywords[@]}"
do
	regexp="$regexp$key|"
done
regexp=`echo $regexp | sed 's/\(.*\)|/\1/'`
regexp="$regexp)$other"
print_debug "[*] Regexp used --> $regexp"

# For all dump files, chek for all the keywords
# by generating a custom regular expression
print_debug "[*] List all files: $dumps_files"
for file in $dumps_files
do
	# Debug information
	print_debug "[*] Analyzing file $file"
	# Search keywords in every dump files
	egrep $regexp $input'/'$file  | awk '{print $2 " " $3}' > $output/$file.output
done
