import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

font = {
        'weight' : 'normal',
        'size'   : 20
        }

matplotlib.rc('font', **font)

# All possible colors
colors = ["b", "g", "r", "c", "m", "y", "k", "#5900b3", "#004d00", "#ff6600"]
weeks = ["42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"]

def plot_data(_range, data, color_gv, label_name):
    plt.plot(_range, data, color=color_gv, linestyle="-", marker='.',  label=label_name)

plt.ylabel("Incidenza su 1000 persone")
plt.xlabel("Settimane")

_range = range(0, len(weeks));

for f in range(0, len(sys.argv)-1):
    _file = sys.argv[f+1]
    document = pd.read_csv(_file)
    labels = document["Settimana"][0:len(weeks)]
    data = document["Incidenza Totale"][0:len(weeks)]

    if len(weeks) > len(data):
        plot_data(range(0, len(data)), data, colors[f], labels[1][0:4])
    else:
        plot_data(_range, data, colors[f], labels[1][0:4])

plt.legend()
plt.xticks(_range, weeks, rotation="vertical")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.show()
