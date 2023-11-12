import pandas as pd


def df_creator():
    file_name = './Flight_Price_Dataset_Q2.csv'
    data_frame = pd.read_csv(file_name)
    encoded_df = pd.DataFrame()

    mapping = {'zero': 3,
               'one': 2,
               'two_or_more': 1}
    encoded_df['stops_mapping'] = data_frame['stops'].map(mapping)

    mapping = {'Economy': 1,
               'Business': 2}
    encoded_df['class_mapping'] = data_frame['class'].map(mapping)

    mapping = {'Morning': 'morning_departure',
               'Early_Morning': 'early_morning_departure',
               'Evening': 'evening_departure',
               'Night': 'night_departure',
               'Afternoon': 'afternoon_departure',
               'Late_Night': 'late_night_departure'}
    data_frame['departure_mapping'] = data_frame['departure_time'].map(mapping)
    departure_dummies = pd.get_dummies(data_frame['departure_mapping'])

    mapping = {'Morning': 'morning_arrival',
               'Early_Morning': 'early_morning_arrival',
               'Evening': 'evening_arrival',
               'Night': 'night_arrival',
               'Afternoon': 'afternoon_arrival',
               'Late_Night': 'late_night_arrival'}
    data_frame['arrival_mapping'] = data_frame['arrival_time'].map(mapping)
    arrival_dummies = pd.get_dummies(data_frame['arrival_mapping'])

    encoded_df = pd.concat([encoded_df, departure_dummies, arrival_dummies, data_frame['duration'],
                            data_frame['days_left'], data_frame['price']], axis=1)
    return encoded_df


df = df_creator()
print(df.keys())
