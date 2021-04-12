import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib
import matplotlib.pyplot as plt

import itertools
import re
from io import StringIO

matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
font = 20
lw = 1
ms = 7
lw2 = 1

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: array, confusion matrix
    :param classes: list, list of defined classes in the model
    :param normalize: bool, info if cm should be normalized
    :param title: str, title of the chart
    :param cmap: color of the chart
    return: plot of the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=font)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=font)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=font)
    plt.yticks(tick_marks, classes, fontsize=font)
    plt.xlim([-0.5, 0.5 + max(tick_marks)])
    plt.ylim([-0.5, 0.5 + max(tick_marks)])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=font,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=font)
    plt.xlabel('Predicted label', fontsize=font)
    plt.tight_layout()

def pandas_classification_report(report):
    """
    Function return df of classification report
    :param report: array, classification report
    :return: dataframe, classification report
    """
    report = (report.split("accuracy")[0] + report.split("accuracy")[1].split("\n")[1])
    report = re.sub(r" +", " ", report).replace("\n ", "\n").replace("macro avg", "avg/total").split("\nweighted")[0]
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)
    return report_df
    
def accuracy(rf, X_train, y_train):
    """
    Calculates the accuracy of a model.
    :return: accuracy score of a model.
    """
    return accuracy_score(y_train, rf.predict(X_train))

