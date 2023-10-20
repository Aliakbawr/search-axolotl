#https://www.w3schools.com/python/pandas/default.asp
import pandas as pd
df = pd.read_csv('./Flight_Data_Easy.csv')
#if you need to use specific column of a specific row, use this line:)
#print(df.iloc[0,3])