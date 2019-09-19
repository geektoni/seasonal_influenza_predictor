#!/bin/bash
# Parse Wikipedia dumps and get information about specific
# pages, based on custom regexps given by input.
# It use default Wikipedia logs (not ordered)
#
# Written by Giovanni De Toni (2017)
# Email: giovanni.det at gmail.com

# General variables
export input=''
export output=''
export language=''
export regexp=''
export exp=''
export pages=''
export debug=false

# Usage and version information
eval "$(docopts -V - -h - : "$@" <<EOF
Usage: parse_dumps_bare (-k <pages> | -e <exp>) [-i <input>] [-o <output>] [-l <language>] [--debug]

Options:
	-k <pages>		File containing the keywords which will be used.
	-e <exp>		File containing the custom generated regexp.
	-i <input>		Specify input directory (default is ./input).
	-o <output>		Specify output directory (default is ./output).
	--debug			Run with debug option.
	--help			Show help options.
	--version		Print program version.
----
parse_dumps_bare 0.1.0
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

# Set up utility variables
# and debug function
function print_debug {
	if $debug; then echo ${1:-}; fi
}

# Generate regexp given a list of words
function generate_regexp {

	# Set up project id, keywords and other field
	project=$language
	keywords=(`cat ${1:-}`)
	other=' .'

	# Build the regexp
	regexp="$project ("
	for key in "${keywords[@]}"
	do
		regexp="$regexp$key|"
	done
	regexp=`echo $regexp | sed 's/\(.*\)|/\1/'`
	regexp="$regexp)$other"

}

# Check all files for pattern
function parse_files {
	# Debug information
	print_debug "[*] Analyzing file ${1:-}"

	# Get filename
	file_name=`basename ${1:-}`
  file_name="${file_name%.*}"

	# Get date
	log_date=`echo $file_name | tr "-" " " | awk '{print $2 "-" $3}'`

	# Search keywords in every dump files
	zegrep "$regexp" ${1:-} | awk -v date=$log_date '{print $2 "," date "," $3}' >> "$output/$file_name.output"

	# Remove empty file (if zegrep didn't find anything)
	if [ ! -s "$output/$file_name.output" ]; then
		rm "$output/$file_name.output"
	fi

}

# Add default values if -i or -o were not set
if [ -z $input ]; then input='./input'; fi
if [ -z $output ]; then output='./output'; fi
if [ -z $language ]; then language='it'; fi

print_debug "[*] Debug value --> $debug"
print_debug "[*] Input directory value --> $input"
print_debug "[*] Output directory value --> $output"
print_debug "[*] Language selected --> $language"
print_debug "[*] Custom regexp --> $exp"
print_debug "[*] Keywords file --> $pages"

# Check if the output directory exists. If it
# is not present then create it (also with its parents)
if [ ! -d $output ]; then
	mkdir -p $output
fi

# Read the regexp which will be used
if [ -z $exp ]; then
	generate_regexp $pages
else
	regexp=`cat $exp`
fi
print_debug "[*] Regexp used --> $regexp"

# Export methods to make them visible for parallel
export -f parse_files
export -f print_debug

# Get parallel version to manage no-notice option
p_v=`./parallel --version | grep parallel | head -1 | awk '{print $3}'`

# Parse the dump files
if [ $p_v -ge 20141022 ]; then
	ls $input/*.gz | ./parallel --no-notice parse_files
else
	ls $input/*.gz | ./parallel parse_files
fi

# Finally, merge together all the files
for f in $output/*.output
do
	[[ -e $f ]] || break  # handle the case of no files
	print_debug "[*] Merging file $f"
	cat $f >> $output/result.output
	rm $f
done
