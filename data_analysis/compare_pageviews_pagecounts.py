import glob
import os

import argparse

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

def generate_dataframe(country, data="new_data"):

    path = "../data/wikipedia_{}/{}".format(country, data)
    files = [os.path.basename(f) for f in glob.glob(path + "/*.csv", recursive=True)]
    years = [f.split(".")[0] for f in files]
    years.sort()

    df = pd.DataFrame()
    for y in years:

        if int(y) >= 2015 and data != "pageviews" and data != "old_data" and data != "cyclerank_pageviews":
            continue

        tmp = pd.read_csv(path + "/" + y + ".csv")

        df = pd.concat([df, tmp], ignore_index=True)

    # Fill nan values
    df.fillna(0, inplace=True)

    # Sum together all the pageviews
    total_pageviews = df.sum(axis=1).to_frame()
    total_pageviews.rename(columns={0: "pagecounts_unorm"}, inplace=True)
    total_pageviews.reset_index(inplace=True)
    total_pageviews["week"] = df["Week"]

    # Remove frames with zero counts
    indexes = total_pageviews[total_pageviews.pagecounts_unorm == 0].index
    total_pageviews.drop(indexes, inplace=True)
    total_pageviews.reset_index(inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    total_pageviews["pagecounts"] = scaler.fit_transform(total_pageviews["pagecounts_unorm"].values.reshape(-1,1))

    return total_pageviews

def get_label(country):
    if country == "italy":
        return "Italian"
    elif country == "germany":
        return "German"
    else:
        return "Dutch"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="pageviews")
    parser.add_argument("--legend", action="store_true", default=False)

    args = parser.parse_args()

    countries = ["italy", "germany", "netherlands"]

    # Generate figure
    fig = plt.figure(figsize=(8,4))

    # Set the weeks
    step = 20
    weeks = []
    max_weeks = 0
    total_observations = 0

    for c in countries:

        df = generate_dataframe(c, args.data)

        # Plot the data
        sns.lineplot(data=df["pagecounts"], label=get_label(c), legend=False)

        # Set the weeks we need to plot
        if max_weeks < len(df["week"]):
            weeks=[]
            counter=0
            for e in df["week"].to_list():
                if counter%step == 0:
                    weeks.append(e)
                counter += 1
            max_weeks = len(df["week"])

        # Set the max number of observations
        total_observations = len(df["pagecounts"]) if len(df["pagecounts"]) > total_observations else total_observations

    # Print the xticks
    plt.xticks(np.arange(0, total_observations, step=step), weeks, rotation=90)

    # Shrink current axis by 20%
    if args.legend:
        ax = fig.axes[0]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    #plt.show()
    plt.savefig("pageviews-numerosity-{}.png".format(args.data), dpi=300, bbox_inches='tight')
