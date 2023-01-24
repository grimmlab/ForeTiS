import sklearn
import numpy as np


def get_evaluation_report(y_true: np.array, y_pred: np.array, prefix: str = '', current_model_name: str = None) -> dict:
    """
    Get values for common evaluation metrics

    :param y_true: true values
    :param y_pred: predicted values
    :param prefix: prefix to be added to the key if multiple eval metrics are collected
    :param current_model_name: name of the current model according to naming of .py file in package model

    :return: dictionary with common metrics
    """
    if len(y_pred) == (len(y_true) - 1):
        print('y_pred has one element less than y_true (e.g. due to batch size config) -> dropped last element')
        y_true = y_true[:-1]
    if current_model_name is not None and 'es' in current_model_name:
        if any(np.isnan(y_pred)):
            mask = ~np.isnan(y_pred)
            y_pred = y_pred[mask]
            y_true = y_true.array[mask]
    eval_report_dict = {
        prefix + 'mse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred),
        prefix + 'rmse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
        prefix + 'r2_score': sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred),
        prefix + 'explained_variance': sklearn.metrics.explained_variance_score(y_true=y_true, y_pred=y_pred)
    }

    return eval_report_dict
