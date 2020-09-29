import pandas as pd
df=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.head())
print(df.columns)
pd.options.display.max_columns=6
print(df.describe())