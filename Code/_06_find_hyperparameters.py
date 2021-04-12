import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

"""
Randomised search for random forest hyperparameters.
"""


def search_hyperparameters(simulation_folder, method='RF', mode='classification'):
    """
    Function for searching best hyperparameters for random forest algorithm
    :param simulation_folder: str, name of subfolder for given data set
    :return: none, hyperparameters are saved down as side effect
    """

    Start = datetime.now()
    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data")
    path_to_characteristics_data = os.path.join(path_to_save, simulation_folder, "Characteristics")
    path_to_hyperparameters = os.path.join(project_directory, "Models", simulation_folder, method, 'Model')
    if not os.path.exists(path_to_hyperparameters):
        os.makedirs(path_to_hyperparameters)
    X_train = np.load(os.path.join(path_to_characteristics_data, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path_to_characteristics_data, "y_train.npy"), allow_pickle=True)

    if method == 'RF':    
        if mode == 'classification':        
            random_params = {'n_estimators': [int(x) for x in range(100, 1001, 100)],
                              'criterion': ["gini", "entropy"],
                              'max_features': ['log2', 'sqrt', None],
                              'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4],
                              'bootstrap': [True, False]
                              }
            box = RandomForestClassifier()
        elif mode == 'regression':
            random_params = {'n_estimators': [int(x) for x in range(100, 1001, 100)],
                              'criterion': ["mae"],
                              'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4],
                              }
            box = RandomForestRegressor()
    elif method == 'GB':  
        if mode == 'classification':  
            random_params = {'n_estimators': [int(x) for x in range(100, 1001, 100)],
                              'max_features': ['log2', 'sqrt', None],
                              'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4],
                              'criterion': ['friedman_mse']
                              }
            box = GradientBoostingClassifier()
        elif mode == 'regression':
            random_params = {'n_estimators': [int(x) for x in range(100, 1001, 100)],
                              'max_features': ['log2', 'sqrt', None],
                              'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4],
                              'criterion': ['friedman_mse']
                              }
            box = GradientBoostingRegressor()
    elif method == 'XGB':
        if mode == 'classification':
            random_params = {'eta': list(np.arange(0.3, 1.1, 0.1)),
                              'max_depth': [int(x) for x in range(1, 11)],
                              'min_child_weight': [int(x) for x in range(1, 11)],
                              'gamma': [int(x) for x in range(0, 11)],
                              'base_score': [0.5, 1],
                              'eval_metric': ['merror', 'auc'],
                              }
            box = XGBClassifier()
        elif mode == 'regression':
            random_params = {'eta': list(np.arange(0.3, 1.1, 0.1)),
                              'max_depth': [int(x) for x in range(1, 11)],
                              'min_child_weight': [int(x) for x in range(1, 11)],
                              'gamma': [int(x) for x in range(0, 11)],
                              'base_score': [1],
                              'eval_metric': ['mae'],
                              }
            box = XGBRegressor()
    # Random search of parameters, using 10 fold cross validation,
    # search across 100 different combinations, and use all available cores
    box_random = RandomizedSearchCV(estimator=box, param_distributions=random_params, n_iter=100, cv=10,
                                   verbose=2, random_state=42, n_jobs=-1)
    # Fit the random search model
    box_random.fit(X_train, y_train)

    print(box_random.best_params_)
    with open(os.path.join(path_to_hyperparameters, "hyperparameters.json"), 'w') as fp:
        json.dump(box_random.best_params_, fp)
    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_hyperparameters, "time_for_searching.csv"))
    print(ExecutedTime)


if __name__ == "__main__":

    search_hyperparameters(simulation_folder="T2_subtask_1D", method='XGB', mode='classification')
