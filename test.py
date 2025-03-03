import csv

somedict = dict(raymond='red', rachel='blue', matthew='green')
with open('mycsvfile.csv','w') as f:
    w = csv.writer(f)
    w.writerow(somedict.keys())
    w.writerow(somedict.values())

# import pandas
import pandas as pd
# read csv
data = pd.read_csv('mycsvfile.csv')
# Convert the DataFrame to a Dictionary
data_dict = data.to_dict(orient='records')
print(data_dict)