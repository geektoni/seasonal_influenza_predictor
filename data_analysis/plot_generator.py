import sys
import pandas as pd
import matplotlib.pyplot as plt

colors = ["b", "g", "r", "c", "m", "y", "k"]
weeks = ["42", "43", "44", "45", "46", "47", "48", "49",
"50", "51", "52", "01", "02", "03",
"04", "05", "06", "07", "09", "10", "11", "12", "13", "14", "15", "16", "17"]

def plot_data(_range, data, color, label_name):
    #plt.plot(_range, data, color+"o")
    plt.plot(_range, data, color+"-", label=label_name)

plt.ylabel("Incidenza su 1000 persone")
plt.xlabel("Settimane")

_range = range(1, 29);

for f in range(1, len(sys.argv)):
    _file = sys.argv[f]
    document = pd.read_csv(_file)
    labels = document["Settimana"]
    data = document["Incidenza Totale"]
    plot_data(_range, data, colors[f], labels[1][0:4])

plt.legend()
plt.xticks(_range, weeks, rotation="vertical")
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.show()
