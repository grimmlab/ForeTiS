import os
import inspect
import importlib
import pandas as pd
import torch
import tensorflow as tf
import random
import numpy as np
from sklearn.model_selection import ShuffleSplit, TimeSeriesSplit
import ForeTiS.model

def get_list_of_implemented_models() -> list:
    """
    Create a list of all implemented models based on files existing in 'model' subdirectory of the repository.
    """
    # Assumption: naming of python source file is the same as the model name specified by the user
    if os.path.exists('../model'):
        model_src_files = os.listdir('../model')
    elif os.path.exists('model'):
        model_src_files = os.listdir('model')
    elif os.path.exists('ForeTiS/model'):
        model_src_files = os.listdir('ForeTiS/model')
    else:
        model_src_files = [model_file + '.py' for model_file in ForeTiS.model.__all__]
    model_src_files = [file for file in model_src_files if file[0] != '_']
    return [model[:-3] for model in model_src_files]


def get_mapping_name_to_class() -> dict:
    """
    Get a mapping from model name (naming in package model without .py) to class name.

    :return: dictionary with mapping model name to class name
    """
    if os.path.exists('../model'):
        files = os.listdir('../model')
    elif os.path.exists('model'):
        files = os.listdir('model')
    elif os.path.exists('ForeTiS/model'):
        files = os.listdir('ForeTiS/model')
    else:
        files = [model_file + '.py' for model_file in ForeTiS.model.__all__]
    modules_mapped = {}
    for file in files:
        if file not in ['__init__.py', '__pycache__']:
            if file[-3:] != '.py':
                continue

            file_name = file[:-3]
            module_name = 'ForeTiS.model.' + file_name
            for name, cls in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
                if cls.__module__ == module_name:
                    modules_mapped[file_name] = cls
    return modules_mapped

def set_all_seeds(seed: int=42):
    """
    Set all seeds of libs with a specific function for reproducibility of results

    :param seed: seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def get_indexes(df: pd.DataFrame, datasplit: str, folds: int, seasonal_valtest, valtest_seasons: int,
                seasonal_periods: int, val_set_size_percentage: int):
    """
    Get the indexes for cv

    :param df: data that should be splited
    :param datasplit: splitting method
    :param folds: number of folds of the hyperparameter optimization
    :param valtest_seasons: the number of seasons to be used for validation and testing when seasonal_valtest is True
    :param seasonal_periods: how many datapoints one season has
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test

    :return: train and test indexes
    """
    train_indexes = []
    test_indexes = []
    if datasplit == 'timeseries-cv':
        if seasonal_valtest:
            for fold in range(folds):
                train_df_index = df.iloc[:-((folds-fold) * valtest_seasons * seasonal_periods)].index
                train_df_indexes = []
                test_df_index = df.iloc[-((folds-fold) * valtest_seasons * seasonal_periods):].index
                test_df_indexes = []

                for i in range(len(train_df_index)):
                    train_df_indexes.append(df.index.get_loc(train_df_index[i]))
                train_indexes.append(np.array(train_df_indexes))
                for i in range(len(test_df_index)):
                    test_df_indexes.append(df.index.get_loc(test_df_index[i]))
                test_indexes.append(np.array(test_df_indexes))
        else:
            splitter = TimeSeriesSplit(n_splits=folds)
            for train_index, test_index in splitter.split(df):
                train_indexes.append(train_index)
                test_indexes.append(test_index)
    if datasplit == 'cv':
        splitter = ShuffleSplit(n_splits=folds, test_size=val_set_size_percentage * 0.01, random_state=0)
        for train_index, test_index in splitter.split(df):
            train_indexes.append(train_index)
            test_indexes.append(test_index)

    return train_indexes, test_indexes



