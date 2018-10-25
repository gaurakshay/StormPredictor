"""
CS 5033 Machine Learning
Solution to Problem 5 of HW 4
Author: Akshay Gaur, 113294004
"""
from time import time
from os.path import dirname, join
from numpy.random import shuffle
from numpy import append, arange, array, genfromtxt, min, square, sum as npsum, where
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plot

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

def preprocess(hw_data):
    """
    Impute NaNs, convert m/s to knots, add classification column.
    """
    wind_index = 20
    ms_to_knot = 1.943844492440605
    # I chose median to impute because I wanted the values from
    # within the dataset and not artificially created.
    hw_data = impute_data(hw_data, 'median')
    hw_data = convert_to_knots(hw_data, wind_index, ms_to_knot)
    hw_data = add_classification(hw_data, wind_index)
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
    return brier_score * (10**14)

def get_bs_bss(model, data, class_indx, severe_percentage):
    """
    With fitted model, predict classification with data provided
    and calculate bs and bss.
    """
    prediction = model.predict_proba(data[:, :class_indx])
    # Probability of the wind being severe is the probability for the
    # class being "1" in dct.classes_ which is at index 1.
    prediction = prediction[:, 1]
    brier_score = get_bs(data[:, class_indx], prediction)
    bs_climo = get_bs(data[:, class_indx], severe_percentage)
    brier_skill_score = (bs_climo - brier_score) / bs_climo
    return brier_score, brier_skill_score

def classify(model, hw_data, class_indx):
    """
    Generate train and validation data, fit the model on
    train data and then use it to calculate bs and bss
    for validation data.
    """
    train_data, validate_data = get_train_validate(hw_data)
    model.fit(train_data[:, :class_indx], train_data[:, class_indx])
    severe_percentage = get_severe_percentage(train_data, class_indx)
    bs, bss = get_bs_bss(model, validate_data, class_indx, severe_percentage)
    return bs, bss

def start_classification(hw_data, no_of_trees, tree_depth):
    """
    Run the model 30 times with given hyperparameters to get the
    average of bs and bss.
    """
    sev_indx = 21
    bs = []
    bss = []
    for _ in range(2):
        gbc = GradientBoostingClassifier(loss='exponential',
                                         n_estimators=no_of_trees,
                                         max_depth=tree_depth)
        bs_, bss_ = classify(gbc, hw_data, sev_indx)
        bs.append(bs_)
        bss.append(bss_)
    avg_bs = npsum(bs) / len(bs)
    avg_bss = npsum(bss) / len(bss)
    print("Avg BS: {}, Avg BSS: {}".format(avg_bs, avg_bss))
    return avg_bs, avg_bss

def find_min(avg_bss):
    """
    find the index of the row and the column
    of the minimum value of the 2d array
    """
    x = array(avg_bss)
    y = where(x == min(x))
    return y[0][0], y[1][0]

def create_color_lot(x_tick_labels, y_tick_labels, data, xlabel, ylabel, colorlabel, title):
    """
    Generate 2D plot for the given data and labels
    """
    fig, ax = plot.subplots()
    heatmap = ax.pcolor(data)
    colorbar = plot.colorbar(heatmap)
    colorbar.set_label(colorlabel, rotation=90)

    ax.set_xticks(arange(len(x_tick_labels)) + 0.5, minor=False)
    ax.set_yticks(arange(len(y_tick_labels)) + 0.5, minor=False)

    ax.set_xticklabels(x_tick_labels, minor=False)
    ax.set_yticklabels(y_tick_labels, minor=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plot.title(title)
    plot.show()
    # plot.savefig(title+".png")
    # fig.clf()
    # fig.clear()
    # ax.cla()
    # ax.clear()
    # plot.cla()
    # plot.clf()
    # plot.close()

def test_classify(model, hw_data, test_data, class_indx):
    """
    Generate train and validation data, fit the model on
    train data and then use it to calculate bs and bss
    for validation data.
    """
    hw_data = hw_data[:10000, :]
    model.fit(hw_data[:, :class_indx], hw_data[:, class_indx])
    severe_percentage = get_severe_percentage(hw_data, class_indx)
    bs, bss = get_bs_bss(model, test_data, class_indx, severe_percentage)
    return bs, bss


def start_grid_search():
    # no_of_trees = [100, 200, 300, 400, 500, 600, 700, 800, 1200, 1500]
    # tree_depth = [3, 4, 5, 6, 7, 8, 9, 10, 13, 20]
    no_of_trees = [100, 200]
    tree_depth = [3, 4]

    hw_data = get_ndarray()
    hw_data = preprocess(hw_data)
    test_data = hw_data[-2000:, :]
    avg_bs = [[] for _ in range(len(no_of_trees))]
    avg_bss = [[] for _ in range(len(no_of_trees))]
    i = 0
    for num in no_of_trees:
        for depth in tree_depth:
            print("="*50)
            print("Number of trees: {}"
                  "\tMax depth: {}".format(num, depth))
            avg_bs_, avg_bss_ = start_classification(hw_data, num, depth)
            avg_bs[i].append(avg_bs_)
            avg_bss[i].append(avg_bss_)
            print("="*50)
        i += 1
    create_color_lot(tree_depth, no_of_trees, avg_bs,
                     "Depth of tree", "Number of trees",
                     "Brier Score (10^-14)",
                     "Brier Score for Depth of tree and Number of trees")
    create_color_lot(tree_depth, no_of_trees, avg_bss,
                     "Depth of tree", "Number of trees",
                     "Brier Skill Score",
                     "Brier Skill Score for Depth of tree and Number of trees")
    
    print("="*50)
    i, j = find_min(avg_bss)
    print("Best model is for # of trees: {}"
          "and max depth: {}".format(no_of_trees[i], tree_depth[j]))

    model = GradientBoostingClassifier(loss='exponential',
                                       n_estimators=no_of_trees[i],
                                       max_depth=tree_depth[j])
    bs, bss = test_classify(model, hw_data, test_data, 21)
    print("Testing BS for best model: {}"
          "\nTesting BSS for best model: {}".format(bs, bss))
    print("="*50)

if __name__ == '__main__':
    # The algorithm selected for this problem is:
    # Gradient Boosting Tree classification.
    # Hyperparameters selected for this task are:
    # 1.    Number of trees
    # 2.    Depth of trees
    print("*"*70)
    print("Grid Search\n"
          "Algorithm: Gradient Boosting Tree classification\n"
          "Hyperparameters selected: Number of trees and depth of trees\n")
    print("*"*70)
    start_grid_search()
