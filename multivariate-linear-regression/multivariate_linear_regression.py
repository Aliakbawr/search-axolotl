import pandas as pd

def df_creator():
    file_name = './Flight_Price_Dataset_Q2.csv'
    data_frame = pd.read_csv(file_name)
    mapping = {'zero': 3,
               'one': 2,
               'two_or_more': 1}

    data_frame['stops_mapping'] = data_frame['stops'].map(mapping)
    mapping = {'Economy': 1,
               'Business': 2}
    data_frame['class_mapping'] = data_frame['class'].map(mapping)

    departure_dummies = pd.get_dummies(data_frame['departure_time'])
    arrival_dummies = pd.get_dummies(data_frame['arrival_time'])
    encoded_df = pd.concat([data_frame, departure_dummies, arrival_dummies], axis=1)
    return encoded_df


df = df_creator()
