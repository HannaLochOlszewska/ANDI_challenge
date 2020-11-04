import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

"""
Functions for generating models' statistics.
"""                                               


def mae(rf, X_train, y_train):
    """
    Calculates the accuracy of a model.
    :return: accuracy score of a model.
    """
    return mean_absolute_error(y_train, rf.predict(X_train))


def imp_df(column_names, importances):
    """
    Transforms features importances into dataframe object.
    :return: DataFrame
    """
    df = pd.DataFrame({'feature': column_names, 'feature_importance': importances}).sort_values('feature_importance',
                                                                                                ascending = False).reset_index(drop = True)
    return df


def drop_col_feat_imp(model, X_train, y_train, random_state=42):
    """
    Function calculating drop column importances of a model.
    :return: DataFrame with drop column feature importances
    """
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis=1), y_train)
        importances.append(benchmark_score - drop_col_score)

    importances_df = imp_df(X_train.columns, importances)
    return importances_df
