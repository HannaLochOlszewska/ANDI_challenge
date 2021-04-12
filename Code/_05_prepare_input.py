import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
Splitting data for train/test set.
"""

def split_data(simulation_folder, mode='classification'):
    """
    Function for spliting data into test and train set
    :param simulation_folder: str, name of subfolder for given data set
    :param target_file: str, name of file with targets (y)
    :param mode: str, if 'classification' we target at the model, if 'regression' - alpha value
    :return: none, files with data are saved down as side effect
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", simulation_folder)

    path_to_characteristics_data = os.path.join(path_to_save, "Characteristics")
    file_with_characteristics = os.path.join(path_to_characteristics_data, "characteristics.csv")
    characteristics_data = pd.read_csv(file_with_characteristics)
#    results_data = pd.read_csv(os.path.join(path_to_save, target_file))
    cols = characteristics_data.columns.drop(['file','Alpha','motion'])
    
    if mode=='classification':    
        X = characteristics_data.loc[:, cols]
        y = characteristics_data['motion']
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
        np.save(os.path.join(path_to_characteristics_data, 'classes.npy'), labelencoder.classes_)
    elif mode=='regression':
        X = characteristics_data.loc[:, cols]
        y = characteristics_data['Alpha']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)

    np.save(os.path.join(path_to_characteristics_data, "X_data.npy"), X)
    np.save(os.path.join(path_to_characteristics_data, "y_data.npy"), y)
    np.save(os.path.join(path_to_characteristics_data, "X_train.npy"), X_train)
    np.save(os.path.join(path_to_characteristics_data, "X_test.npy"), X_test)
    np.save(os.path.join(path_to_characteristics_data, "y_train.npy"), y_train)
    np.save(os.path.join(path_to_characteristics_data, "y_test.npy"), y_test)
    
    return

if __name__ == "__main__":

    split_data(simulation_folder="T2_subtask_1D", mode='classification')
    split_data(simulation_folder="T2_subtask_2D", mode='classification')
