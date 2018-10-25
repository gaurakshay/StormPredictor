"""
CS 5033 Machine Learning
Solution to Problem 2 of HW 4
Author: Akshay Gaur, 113294004
"""
from os.path import dirname, join
from numpy.random import shuffle
from numpy import append, genfromtxt, square, sum as npsum
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def get_ndarray():
    """
    Get the data in ndarray form.
    """
    # Specify the filepath.
    pwd = dirname(__file__)
    filepath = join(pwd, 'cs5033_fall2017_assign04_data.csv')
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

def get_severe_percentage(train_data, index):
    """
    Get the probability of wind being severe.
    """
    severe_data = train_data[train_data[:, index] == 1]
    severe_percentage = len(severe_data)/len(train_data)
    return severe_percentage

def get_bs(value, prediction):
    """
    Get mean squared prediction error
    """
    err_pred = value - prediction
    brier_score = npsum(square(err_pred)) / len(value)
    return brier_score

def get_bss(model, data, class_indx, severe_percentage):
    """
    With fitted model, predict classification with data provided
    and calculate bs and bss.
    """
    prediction = model.predict_proba(data[:, :class_indx])
    # Probability of the wind being severe is the probability for the
    # class being "1" in dct.classes_ which is at index 1.
    prediction = prediction[:, 1]
    brier_score = get_bs(data[:, class_indx], prediction)
    print("BS score is {}".format(brier_score))
    bs_climo = get_bs(data[:, class_indx], severe_percentage)
    brier_skill_score = (bs_climo - brier_score) / bs_climo
    print("BSS score is {}".format(brier_skill_score))
    # return brier_skill_score

def classify(model, hw_data, test_data, class_indx):
    """
    Generate test and validation data, fit the model on
    test data and then use it to calculate bs and bss
    for validation and test data.
    """
    # model = DecisionTreeClassifier(criterion="entropy")
    train_data, validate_data = get_train_validate(hw_data)
    model.fit(train_data[:, :class_indx], train_data[:, class_indx])
    severe_percentage = get_severe_percentage(train_data, class_indx)
    print("Calculating BS, BSS for validation data...")
    get_bss(model, validate_data, class_indx, severe_percentage)
    print("Calculating BS, BSS for test data...")
    get_bss(model, test_data, class_indx, severe_percentage)

def start_classification():
    """
    Preprocess the data, generate different models for classification.
    """
    wind_indx = 20
    sev_indx = 21
    ms_to_knot = 1.943844492440605
    hw_data = get_ndarray()
    hw_data = impute_data(hw_data, 'median')
    hw_data = convert_to_knots(hw_data, wind_indx, ms_to_knot)
    hw_data = add_classification(hw_data, wind_indx)
    test_data = hw_data[-2000:, :]
    print("*"*70)
    print("Decision Tree Classifier")
    print("*"*70)
    dtc = DecisionTreeClassifier(criterion="entropy")
    classify(dtc, hw_data, test_data, sev_indx)
    print("*"*70)
    print("Random Forest Classifier")
    print("*"*70)
    rfc = RandomForestClassifier(n_estimators=500, criterion="entropy")
    classify(rfc, hw_data, test_data, sev_indx)
    print("*"*70)
    print("Gradient Boosting Classifier")
    print("*"*70)
    gbc = GradientBoostingClassifier(loss='exponential', n_estimators=500)
    classify(gbc, hw_data, test_data, sev_indx)

if __name__ == '__main__':
    start_classification()
