#!/bin/bash
# Parse Wikipedia dumps and get information of specific
# pages, based on custom generated regexps.
#
# Written by Giovanni De Toni (2017)

# General variables
export input=''
export output=''
export debug=false

# Usage and version information
eval "$(docopts -V - -h - : "$@" <<EOF
Usage: parse_dumps [-i <input>] [-o <output>] [--debug]

Options:
	-i <input>		Specify input directory (default is ./input)
	-o <output>		Specify output directory (default is ./output)
	--debug			Run with debug option.
	--help			Show help options.
	--version		Print program version.
----
parse_dumps 0.1.0
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

# Check all files for pattern
function parse_files {
	# Debug information
	print_debug "[*] Analyzing file ${1:-}"

	# Get filename
	file_name=`basename ${1:-}`

	# Search keywords in every dump files
	zegrep "$regexp" ${1:-} | awk '{print $2 "," $3 "," $4 "," $5}' >> "$output/$file_name.output"

	# Remove empty file (if zegrep didn't find anything)
	if [ ! -s "$output/$file_name.output" ]; then
		rm "$output/$file_name.output"
	fi

}

# Add default values if -i or -o were not set
if [ -z $input ]; then input='./input'; fi
if [ -z $output ]; then output='./output'; fi

print_debug "[*] Debug value --> $debug"
print_debug "[*] Input directory value --> $input"
print_debug "[*] Output directory value --> $output"

# Set keywords, project and an other
# section to add other constraints.
project="it"
keywords=(
	"Centro_europeo_per_la_prevenzione_e_il_controllo_delle_malattie"
	"Epidemia"
	"Febbre"
	"Influenza"
	"Influenza_Aviaria"
	"Influenza_spagnola"
	"Influenza_suina"
	"Influenzavirus_A"
	"Influenzavirus_A_sottotipo_H1N1"
	"Oseltamivir"
	"Pagina_Principale"
	"Pandemia"
	"Pandemia_influenzale"
	"Raffreddore_comune"
	"Vaccino"
	"Vaccino_antinfluenzale"
)
other="."

# Check if the output directory exists
if [ ! -d $output ]; then
	mkdir $output
fi

# Generate the regexp concatenating
# all the keyword
export regexp="$project ("
for key in "${keywords[@]}"
do
	regexp="$regexp$key|"
done
regexp=`echo $regexp | sed 's/\(.*\)|/\1/'`
regexp="$regexp) $other"
print_debug "[*] Regexp used --> $regexp"


# Export methods to make them visible for parallel
export -f parse_files
export -f print_debug

# Get parallel version to manage no-notice option
p_v=`parallel --version | grep parallel | head -1 | awk '{print $3}'`

# Parse the dump files
if [ $p_v -ge 20141022 ]; then
	ls $input/*.gz | parallel --no-notice parse_files
else
	ls $input/*.gz | parallel parse_files
fi

# Finally, merge together all the files
for f in $output/*.gz.output
do
	[[ -e $f ]] || break  # handle the case of no files
	print_debug "[*] Merging file $f"
	cat $f >> $output/result.output
	rm $f
done
