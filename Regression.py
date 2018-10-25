"""
CS 5033 Machine Learning
Solution to Problem 3 of HW 4
Author: Akshay Gaur, 113294004
"""
from os.path import dirname, join
from numpy.random import shuffle
from numpy import absolute, append, genfromtxt, sqrt, square, sum as npsum
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def get_ndarray():
    """
    Get the data in ndarray form.
    """
    # Specify the filepath.
    filepath = join(dirname(__file__), 'cs5033_fall2017_assign04_data.csv')
    # generate ndarray from the data file.
    hw_data = genfromtxt(filepath, delimiter=',', skip_header=1)
    return hw_data

def impute_data(hw_data, strat):
    """
    Resolve the NaNs in the data using strategy strat.
    """
    imputer = Imputer(strategy=strat)
    hw_data = imputer.fit_transform(hw_data)
    return hw_data

def convert_to_knots(hw_data, index, factor):
    """
    Convert meter per second to knots.
    """
    hw_data[:, index] = hw_data[:, index] * factor
    return hw_data

def add_classification(hw_data, index):
    """
    Add a column at the end of the data to capture wind severity.
    0 if wind speed is < 50
    1 if wind speed is >= 50
    """
    sev_wind = hw_data[:, index] >= 50
    sev_wind = sev_wind.astype(int)
    sev_wind = sev_wind.reshape(len(sev_wind), 1)
    hw_data = append(hw_data, sev_wind, axis=1)
    return hw_data

def get_train_validate(hw_data):
    """
    Generate training and validation data sets.
    """
    hw_data = hw_data[:10000, :]
    shuffle(hw_data)
    train_data = hw_data[:7500, :]
    validate_data = hw_data[-2500:, :]
    return train_data, validate_data

def get_mae(value, prediction):
    """
    Get Mean Absolute Error
    """
    err = absolute(value - prediction)
    mae = npsum(err) / len(value)
    return mae

def get_rmse(value, prediction):
    """
    Get Root Mean Squared Error
    """
    err = value - prediction
    mean_sq = sum(square(err))
    rmse = sqrt(mean_sq / len(value))
    return rmse


def get_err(model, data, class_indx):
    """
    With fitted model, predict wind speed with data provided
    and calculate MAE and RMSE.
    """
    prediction = model.predict(data[:, :class_indx])
    mae = get_mae(data[:, class_indx], prediction)
    print("MAE is {}".format(mae))
    rmse = get_rmse(data[:, class_indx], prediction)
    print("RMSE is {}".format(rmse))

def classify(model, hw_data, test_data, class_indx):
    """
    Generate test and validation data, fit the model on
    test data and then use it to calculate MAE and RMSE
    for validation and test data.
    """
    train_data, validate_data = get_train_validate(hw_data)
    model.fit(train_data[:, :class_indx], train_data[:, class_indx])
    print("Calculating MAE, RMSE for validation data...")
    get_err(model, validate_data, class_indx)
    print("Calculating MAE, RMSE for test data...")
    get_err(model, test_data, class_indx)

def start_regression():
    """
    Preprocess the data, generate different models for classification.
    """
    wind_indx = 20
    ms_to_knot = 1.943844492440605
    hw_data = get_ndarray()
    hw_data = impute_data(hw_data, 'median')
    hw_data = convert_to_knots(hw_data, wind_indx, ms_to_knot)
    hw_data = add_classification(hw_data, wind_indx)
    test_data = hw_data[-2000:, :]
    print("*"*70)
    print("Decision Tree Regressor")
    print("*"*70)
    dtr = DecisionTreeRegressor()
    classify(dtr, hw_data, test_data, wind_indx)
    print("*"*70)
    print("Random Forest Regressor")
    print("*"*70)
    rfr = RandomForestRegressor(n_estimators=500)
    classify(rfr, hw_data, test_data, wind_indx)
    print("*"*70)
    print("Gradient Boosting Regressor")
    print("*"*70)
    gbr = GradientBoostingRegressor(n_estimators=500)
    classify(gbr, hw_data, test_data, wind_indx)

if __name__ == '__main__':
    start_regression()
