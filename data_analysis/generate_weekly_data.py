import fileinput
import pandas as pd

# Set up a dictionary
all_data={}

# Read from standard input
for line in fileinput.input():

    # Split the line given
    result = str.split(line)

    # Set up an empty list if the key
    # is null
    if all_data.get(result[0], []) == []:
        all_data[result[0]] = [0 for x in range(53)]

    # Sum the visit counter
    all_data[result[0]][int(result[1])] += int(result[2]);

# Print all the data
#for key, value in all_data.iteritems():
#    print(key)
#    print(value)

# Generate a pandas dataframe with all the data
df = pd.DataFrame(all_data);
print(df)

# Save it to file
df.to_csv("result.csv", index_label="Week")
