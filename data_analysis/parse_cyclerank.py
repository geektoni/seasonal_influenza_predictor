#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine learning model which uses Wikipedia data to predicts ILI levels.

Usage:
  parse_cyclerank [--base_directory=<base_directory>] [--language=<language>] [--word=<word>] [--loop_size=<loop_size>] [--output=<output>]

  <base_directory>
  <language>
  <word>
  <loop_size>
  -h, --help        Print this help message
"""

import os
import glob
import re
import pprint
from docopt import docopt
from tqdm import tqdm

if __name__ == '__main__':

    # Parse the command line
    args = docopt(__doc__)

    directory_base = "../data/cyclerank/"
    language="de"
    word="Influenza"
    loop_size=3
    output ="./"

    if args["--base_directory"]:
        directory_base = args["--base_directory"]
    if args["--language"]:
        language = args["--language"]
    if args["--word"]:
        word =args["--word"]
    if args["--loop_size"]:
        loop_size = args["--loop_size"]
    if args["--output"]:
        output = args["--output"]

    data_files = [os.path.basename(x) for x in glob.glob(directory_base+language+"*")]

    files = [file_ for file_ in data_files if not "scores.txt" in file_ and "fexpe.{}".format(word) in file_ and ".{}.".format(loop_size) in file_]

    print("[*] File list:")
    pp = pprint.PrettyPrinter(indent=4)
    files.sort()
    pp.pprint(files)

    print("[*] Generating list of unique words:")
    page_list = []
    for f in tqdm(files):
        with open(directory_base+f, "r") as opened:
            lines = opened.readlines()
            for l in lines:
                splitted = l.rsplit("):", 1)
                page = splitted[0].replace("score(", "", 1)
                page_list.append(page)
    page_list = set(page_list)

    with open(output+"keywords_cyclerank_{}.txt".format(language), "w") as output:
        for p in page_list:
            output.write(p+"\n")
