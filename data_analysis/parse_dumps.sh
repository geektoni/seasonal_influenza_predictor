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
	-y <year>			Year that should be analyzed
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
export debug=0
export new_input=0
export input="/mnt/fluiddata/cconsonni/pagecounts/data/output"
export output="./output"
export year="2007"

# Parse command line arguments
while getopts ":i:o:y:dvh" opt; do
	case $opt in
		i)
			export input=$OPTARG
			export new_input=1
      		;;
		o)
			export output=$OPTARG
			;;
		y)
			export year=$OPTARG
			;;
		d)
			export debug=1
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

# Check all files for pattern
function parse_files {
	# Debug information
	print_debug "[*] Analyzing file ${1:-}"

	# Search keywords in every dump files
	zegrep "$regexp" ${1:-} | awk '{print $2 "," $3 "," $4 "," $5}' >> $output/$year.output

	# If the file generated is empty delete everything
	#if [ ! -s $output/$file_name.output ]; then
	#	rm -rf $output/$file_name.output
	#fi
}

print_debug "[*] Debug value --> $debug"
print_debug "[*] Input directory value --> $input"
print_debug "[*] Output directory value --> $output"
print_debug "[*] Year requested --> $year"

# All Wikipedia dump files of a specific year
dumps_files=`ls $input/$year-*/output-*/*.gz`

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

# Check if there is already a file
if [ -f $output/$year.output ]; then
	echo "The file $year.output already exists."
	exit 1
fi

# Generate the regexp concatenating
# all the keyword
export regexp="$project ("
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

export -f parse_files
export -f print_debug

# Workaround to suppress parallel messages
p_v=`parallel --version | grep parallel | head -1 | awk '{print $3}'`

# Ugly way to manage custom input directories
# FIXME
if [ $p_v -ge 20141022 ]; then
	if [ ! $new_input -eq 1 ]; then
		ls $input/$year-*/output-*/*.gz | parallel --no-notice parse_files
	else
		ls $input/*.gz | parallel --no-notice parse_files
	fi
else
	if [ ! $new_input -eq 1 ]; then
		ls $input/$year-*/output-*/*.gz | parallel parse_files
	else
		ls $input/*.gz | parallel parse_files
	fi
fi
