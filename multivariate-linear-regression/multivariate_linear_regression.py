import pandas as pd

#Handling csv file
file_name = 'Flight_Price_Dataset_Q2.csv'
df = pd.read_csv(file_name)
df_1 = df.drop_duplicates(subset=['departure_time'])
print(df_1)
print('================================================')
df_2 = df.drop_duplicates(subset=['stops'])
print(df_2)
print('================================================')
df_3 = df.drop_duplicates(subset=['arrival_time'])
print(df_3)
print('================================================')
df_4 = df.drop_duplicates(subset=['class'])
print(df_4)
print('================================================')
df_5 = df.drop_duplicates(subset=['duration'])
print(df_5)
print('================================================')
df_6 = df.drop_duplicates(subset=['days_left'])
print(df_6)
print('================================================')
df_7 = df.drop_duplicates(subset=['price'])
print(df_7)
print('================================================')


