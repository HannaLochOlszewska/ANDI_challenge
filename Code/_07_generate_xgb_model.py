import json
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

import xgboost as xgb

np.set_printoptions(precision=2)
"""
Generating the gradient boosting model with the pre-saved hyperparameters.
"""

def generate_model(simulation_folder):
    """
    Function for generating model for given scenario and feature based model
    :param simulation_folder: str, name of subfolder for given data set
    :return: none, model is saved down as side effect
    """

    Start = datetime.now()
    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", simulation_folder)
    path_to_characteristics_data = os.path.join(path_to_data, "Characteristics")
    path_to_model = os.path.join(project_directory, "Models", simulation_folder,
                                 "XGB", "Model")
    path_to_hyperparameters = os.path.join(path_to_model, "hyperparameters.json")

    X_train = np.load(os.path.join(path_to_characteristics_data, "X_train.npy"))
    y_train = np.load(os.path.join(path_to_characteristics_data, "y_train.npy"))

    with open(path_to_hyperparameters, 'r') as f:
        param_data = json.load(f)
    model = xgb.XGBRegressor()
    model.set_params(**param_data)
    model.fit(X_train, y_train)   
    joblib.dump(model, os.path.join(path_to_model, 'model.sav'))
    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_model, "time_for_modelling.csv"))
    print(ExecutedTime)

if __name__ == "__main__":
    
    generate_model(simulation_folder="Base_subtask_1D")
    
