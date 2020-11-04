import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
Splitting data for train/test set.
"""

def split_data(simulation_folder, target_file):
    """
    Function for spliting data into test and train set
    :param simulation_folder: str, name of subfolder for given data set
    :param target_file: str, name of file with targets (y)
    :return: none, files with data are saved down as side effect
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", simulation_folder)

    path_to_characteristics_data = os.path.join(path_to_save, "Characteristics")
    file_with_characteristics = os.path.join(path_to_characteristics_data, "characteristics.csv")
    characteristics_data = pd.read_csv(file_with_characteristics)
    results_data = pd.read_csv(os.path.join(path_to_save, target_file))
    characteristics_data = characteristics_data.drop(["file", "motion"], axis=1)
    X = characteristics_data.loc[:, characteristics_data.columns != 'diff_type']
    
    y = results_data['Alpha']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    np.save(os.path.join(path_to_characteristics_data, "X_data.npy"), X)
    np.save(os.path.join(path_to_characteristics_data, "y_data.npy"), y)
    np.save(os.path.join(path_to_characteristics_data, "X_train.npy"), X_train)
    np.save(os.path.join(path_to_characteristics_data, "X_test.npy"), X_test)
    np.save(os.path.join(path_to_characteristics_data, "y_train.npy"), y_train)
    np.save(os.path.join(path_to_characteristics_data, "y_test.npy"), y_test)
    
    return

if __name__ == "__main__":

    split_data(simulation_folder="Base_subtask_1D", target_file='alphas.txt')
