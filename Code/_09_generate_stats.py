import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from rfpimp import permutation_importances
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

from _08_stats import imp_df, drop_col_feat_imp, mae

"""
Module that generates statistics of the trained model.
"""

def generate_stats(simulation_folder):
    """
    Function for generating statistics about model
    :param simulation_folder: str, name of subfolder for given data sets
    :return: none, statistics saved down as side effect
    """

    Start = datetime.now()
    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", simulation_folder)
    path_to_characteristics_data = os.path.join(path_to_data, "Characteristics")
    path_to_scenario = os.path.join(project_directory, "Models", simulation_folder,
                                    "XGB", "Model")
    path_to_stats = os.path.join(path_to_scenario, "Stats")
    if not os.path.exists(path_to_stats):
        os.makedirs(path_to_stats)
    path_to_model = os.path.join(path_to_scenario, "model.sav")
    
    X_train = np.load(os.path.join(path_to_characteristics_data, "X_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(path_to_characteristics_data, "X_test.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path_to_characteristics_data, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path_to_characteristics_data, "y_test.npy"), allow_pickle=True)
    # TODO: fix the save of the data to get variable names from there
    characteristics_data = pd.read_csv(os.path.join(path_to_characteristics_data, "characteristics.csv"))
    model = joblib.load(path_to_model)
    data_type = ["Train", "Test"]
    for dt in data_type:
        X = X_train if dt == "Train" else X_test
        y = y_train if dt == "Train" else y_test
        # Making the Confusion Matrix
        y_pred = model.predict(X)

    ## TODO: mae per class
        print("mae")
        test_train_mae = mean_absolute_error(y, y_pred)
        df = pd.DataFrame({'mae': [test_train_mae]})
        df.to_csv(os.path.join(path_to_stats, "MAE_" + dt + ".csv"))

    # feature importances
    importances = model.feature_importances_
    column_names = characteristics_data.drop(["file", "motion", "diff_type"], axis=1).columns.values
    df = imp_df(column_names, importances)
    df.to_csv(os.path.join(path_to_stats, "Feature_importances.csv"), index=False)
    

    # permutation importances
    X_train_df = pd.DataFrame(X_train, columns=column_names)
    y_train_df = pd.DataFrame(y_train)
    md = clone(model)
    md.fit(X_train_df,y_train_df)
    df = permutation_importances(md, X_train_df, y_train_df, mae)
    df.to_csv(os.path.join(path_to_stats, "Permutation_fi.csv"), index=True)

    # drop column feature importance
    X_train_df = pd.DataFrame(X_train, columns=column_names)
    df = drop_col_feat_imp(model, X_train_df, y_train)
    df.to_csv(os.path.join(path_to_stats, "Drop_column_fi.csv"), index=False)

    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_stats, "time_for_stats_generator.csv"))
    print(ExecutedTime)


if __name__ == "__main__":
    generate_stats(simulation_folder="Base_subtask_1D")
