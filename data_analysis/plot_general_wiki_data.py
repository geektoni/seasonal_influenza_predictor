import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

if __name__ == "__main__":

    path = "../data/general_wikipedia_data"
    filename_legacy = "legacy-page-views"
    filename_current = "total-page-views"
    languages = [ "italian", "german", "dutch", "english"]
    ax = [(0,0), (0,1), (1,0), (1,1)]

    figure, axes = plt.subplots(2, 2, figsize=(14, 8))

    counter=0

    for l in languages:

        legacy = pd.read_csv(path+"/"+filename_legacy+"-"+l+".csv")[["month", "total.desktop-site", "total.mobile-site"]]
        current = pd.read_csv(path + "/" + filename_current + "-" + l + ".csv")[["month", "total.desktop", "total.mobile-app", "total.mobile-web"]]

        current["total.mobile"] = current["total.mobile-app"]+current["total.mobile-web"]
        current = current[["month", "total.desktop", "total.mobile"]]
        legacy = legacy.rename(columns={"total.desktop-site": "total.desktop", "total.mobile-site": "total.mobile"})

        # Concatenate the two lists
        tmp = pd.concat([legacy, current], axis=0, ignore_index=True)
        tmp.fillna(0, inplace=True)

        # Generate the list of weeks
        weeks_split = tmp["month"].str.split("-")
        weeks = []
        for i in range(0,len(weeks_split)):
            if i%10 == 0:
                weeks.append(weeks_split[i][0]+"-"+weeks_split[i][1])

        # Plot the results
        sns.lineplot(data=tmp["total.desktop"], label="desktop", ax=axes[ax[counter]], legend=False, linewidth=2.5)
        sns.lineplot(data=tmp["total.mobile"], label="mobile", ax=axes[ax[counter]], legend=False, linewidth=2.5)
        axes[ax[counter]].set_xticks(np.arange(0, len(weeks_split), step=10))
        axes[ax[counter]].set_xticklabels(weeks, fontdict={"rotation":90})
        axes[ax[counter]].legend(loc="upper right")
        axes[ax[counter]].get_legend().set_title(l)

        counter += 1

    plt.tight_layout()
    #plt.show()
    plt.savefig("variation-of-pageviews.png", dpi=250)