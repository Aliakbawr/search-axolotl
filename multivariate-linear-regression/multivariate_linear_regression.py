import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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


def define_and_normalize_xs():
    a1 = df['stops_mapping']
    a2 = df['class_mapping']
    a3 = df['duration']
    a4 = df['days_left']
    a5 = df['afternoon_departure']
    a5 = a5.astype(int)
    a6 = df['early_morning_departure']
    a6 = a6.astype(int)
    a7 = df['evening_departure']
    a7 = a7.astype(int)
    a8 = df['late_night_departure']
    a8 = a8.astype(int)
    a9 = df['morning_departure']
    a9 = a9.astype(int)
    a10 = df['night_departure']
    a10 = a10.astype(int)
    a11 = df['afternoon_arrival']
    a11 = a11.astype(int)
    a12 = df['early_morning_arrival']
    a12 = a12.astype(int)
    a13 = df['evening_arrival']
    a13 = a13.astype(int)
    a14 = df['late_night_arrival']
    a14 = a14.astype(int)
    a15 = df['morning_arrival']
    a15 = a15.astype(int)
    a16 = df['night_arrival']
    a16 = a16.astype(int)
    # Normalize features
    a1 = (a1 - a1.mean()) / a1.std()
    a2 = (a2 - a2.mean()) / a2.std()
    a3 = (a3 - a3.mean()) / a3.std()
    a4 = (a4 - a4.mean()) / a4.std()
    a5 = (a5 - a5.mean()) / a5.std()
    a6 = (a6 - a6.mean()) / a6.std()
    a7 = (a7 - a7.mean()) / a7.std()
    a8 = (a8 - a8.mean()) / a8.std()
    a9 = (a9 - a9.mean()) / a9.std()
    a10 = (a10 - a10.mean()) / a10.std()
    a11 = (a11 - a11.mean()) / a11.std()
    a12 = (a12 - a12.mean()) / a12.std()
    a13 = (a13 - a13.mean()) / a13.std()
    a14 = (a14 - a14.mean()) / a14.std()
    a15 = (a15 - a15.mean()) / a15.std()
    a16 = (a16 - a16.mean()) / a16.std()
    return np.c_[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]


# Algorithm
def gradiant_descent(X, Y):
    learning_rate = 0.001
    epochs = 2700
    N = Y.size
    coeff = np.random.rand(17)
    past_costs = []
    PAST_COEFF = [coeff]
    for i in range(epochs):
        prediction = np.dot(X, coeff)
        error = prediction - Y
        cost = 1 / (2 * N) * np.dot(error.T, error)
        past_costs.append(cost)
        der = (1 / N) * learning_rate * np.dot(X.T, error)
        coeff = coeff - der
        PAST_COEFF.append(coeff)
    return PAST_COEFF, past_costs


def generate_errors():
    # Predictions
    predictions = np.dot(x_test, coeffi)
    # Mean Squared Error (MSE)
    mse = np.mean((y_test - predictions) ** 2)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # R-squared (R²)
    mean_y = np.mean(y_test)
    ss_total = np.sum((y_test - mean_y) ** 2)
    ss_residual = np.sum((y_test - predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, predictions)
    return mse, rmse, mae, r_squared


def generate_file():
    st = "PRICE = "
    for k in range(16):
        st = st + f' ({coeffi[k]}) * [{df.columns[k]}] +'

    ans = st[:-1]
    t = f'\nTraining Time: {round(end_time - start_time)}s'
    MSE, RMSE, MAE, R_SQUARED = generate_errors()
    errors = (f'\n\nLogs: '
              f'\nMSE: {MSE}'
              f'\nRMSE: {RMSE}'
              f'\nMAE: {MAE}'
              f'\nR2: {R_SQUARED}')

    file = open('[2]-UIAI4021-PR1-Q2.txt.txt', 'w')
    file.write(ans)
    file.write(t)
    file.write(errors)
    file.close()


df = df_creator()
# Define "y"
y = df['price']
x = np.c_[define_and_normalize_xs(), np.ones(270138)]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
start_time = time.time()
past_coeff, past_cost = gradiant_descent(x_train, y_train)
end_time = time.time()
coeffi = past_coeff[-1]

generate_file()
