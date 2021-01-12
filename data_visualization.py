import pandas as pd
import numpy as np
#"Link","Title","Author","Rating Count","Review Count","Rating Value","N pag","1st Pub","series","Genres","Awards"
df = pd.read_csv("./resources/Books.csv")
#print(df['Awards'])

def count_awards(string_award):
    if string_award is not np.NaN:
        string_award = string_award.split(",")
        return len(string_award)
    else:
        return np.NaN



df['Awards'] = df['Awards'].apply(count_awards)
sequence = ["Title","Author","Rating Count","Review Count","Rating Value","N pag","1st Pub","series","Genres","Awards","Link"]
df = df.reindex(columns = sequence)
print(df.head())
df.to_csv('./resources/Books.csv') 