import pandas as pd
import numpy as np


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


# Define "x"s
df = df_creator()
x1 = df['stops_mapping']
x2 = df['class_mapping']
x3 = df['duration']
x4 = df['days_left']
x5 = df['afternoon_departure']
x5 = x5.astype(int)
x6 = df['early_morning_departure']
x6 = x6.astype(int)
x7 = df['evening_departure']
x7 = x7.astype(int)
x8 = df['late_night_departure']
x8 = x8.astype(int)
x9 = df['morning_departure']
x9 = x9.astype(int)
x10 = df['night_departure']
x10 = x10.astype(int)
x11 = df['afternoon_arrival']
x11 = x11.astype(int)
x12 = df['early_morning_arrival']
x12 = x12.astype(int)
x13 = df['evening_arrival']
x13 = x13.astype(int)
x14 = df['late_night_arrival']
x14 = x14.astype(int)
x15 = df['morning_arrival']
x15 = x15.astype(int)
x16 = df['night_arrival']
x16 = x16.astype(int)

# Define "y"
y = df['price']

# Normalize features
x1 = (x1 - x1.mean()) / x1.std()
x2 = (x2 - x2.mean()) / x2.std()
x3 = (x3 - x3.mean()) / x3.std()
x4 = (x4 - x4.mean()) / x4.std()
x5 = (x5 - x5.mean()) / x5.std()
x6 = (x6 - x6.mean()) / x6.std()
x7 = (x7 - x7.mean()) / x7.std()
x8 = (x8 - x8.mean()) / x8.std()
x9 = (x9 - x9.mean()) / x9.std()
x10 = (x10 - x10.mean()) / x10.std()
x11 = (x11 - x11.mean()) / x11.std()
x12 = (x12 - x12.mean()) / x12.std()
x13 = (x13 - x13.mean()) / x13.std()
x14 = (x14 - x14.mean()) / x14.std()
x15 = (x15 - x15.mean()) / x15.std()
x16 = (x16 - x16.mean()) / x16.std()

x = np.c_[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, np.ones(x1.shape[0])]
x_mat = x.shape
print(x_mat)

# Gradiant Descent Algorithm
learning_rate = 0.001
epochs = 20000
N = y.size
coeff = np.random.rand(17)
print('Initial values for coefficients : ', coeff)


# Algorithm
def gradiant_descent(x, y, coeff, epochs, learning_rate):
    past_costs = []
    past_coeff = [coeff]
    for i in range(epochs):
        prediction = np.dot(x, coeff)
        error = prediction - y
        cost = 1 / (2 * N) * np.dot(error.T, error)
        past_costs.append(cost)
        der = (1 / N) * learning_rate * np.dot(x.T, error)
        coeff = coeff - der
        past_coeff.append(coeff)
    return past_coeff, past_costs


past_coeff, past_cost = gradiant_descent(x, y, coeff, epochs, learning_rate)
coeff = past_coeff[-1]
print("final values of coefficients : ", coeff)

# Predictions
predictions = np.dot(x, coeff)

# Mean Squared Error (MSE)
mse = np.mean((y - predictions) ** 2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared (R²)
mean_y = np.mean(y)
ss_total = np.sum((y - mean_y) ** 2)
ss_residual = np.sum((y - predictions) ** 2)
r_squared = 1 - (ss_residual / ss_total)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r_squared}')
