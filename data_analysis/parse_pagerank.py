import pandas as pd
import argparse
import glob
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', type=str, default="it", help="directory where the files are stored")
    parser.add_argument('--top', type=int, default=100)

    args = parser.parse_args()

    min_year = 2007
    max_year = 2019
    top_k = args.top

    total_pages = []

    for f in glob.glob(args.dir+"*results.txt"):

        name = os.path.basename(f)
        year = int(name.split(".")[4].split("-")[0])

        if not (year >= min_year and year <= max_year):
            continue

        with open(f, "r") as file:
            counter = 0
            for line in file:

                if counter >= top_k:
                    break

                data = line.split("\t")

                page_name = data[0][6:-2].replace(" ", "_")
                score = float(data[1])

                total_pages.append(page_name)

                counter += 1

    total_pages = set(total_pages)
    print(len(total_pages))

    with open("result.txt", "w+") as output:

        for name in total_pages:
            output.write(name+"\n")
