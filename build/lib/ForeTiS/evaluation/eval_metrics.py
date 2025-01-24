import sklearn
import numpy as np

def smape(y_true: np.array, y_pred: np.array) -> float:
    """
    Function delivering Symmetric Mean Absolute Percentage Error between prediction and actual values
    :param y_true: actual values
    :param y_pred: prediction values
    :return: sMAPE between prediction and actual values
    """
    return 100 / len(y_true) * np.sum(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))


def mape(y_true: np.array, y_pred: np.array) -> float:
    """
    Function delivering Mean Absolute Percentage Error between prediction and actual values
    :param y_true: actual values
    :param y_pred: prediction values
    :return: MAPE between prediction and actual values
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100  # +0.1 to avoid div by zero

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
        prefix + 'explained_variance': sklearn.metrics.explained_variance_score(y_true=y_true, y_pred=y_pred),
        prefix + 'MAPE': mape(y_true=y_true, y_pred=y_pred),
        prefix + 'sMAPE': smape(y_true=y_true, y_pred=y_pred)
    }

    return eval_report_dict
