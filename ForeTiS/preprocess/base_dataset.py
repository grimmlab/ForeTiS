import pandas as pd
import numpy as np
import os
import warnings
import configparser
from sklearn.model_selection import train_test_split

from .raw_data_functions import custom_resampler, drop_columns, get_one_hot_encoded_df, get_iter_imputer, \
    get_simple_imputer, get_knn_imputer
from . import FeatureAdder


class Dataset:
    """
    Class containing datasets ready for optimization.

    **Attributes**

        - target_column (*str*): the target column for the prediction
        - data_dir (*str*): data directory where the data is stored
        - data (*str*): the dataset that you want to use
        - windowsize_current_statistics (*int*): the windowsize for the feature engineering of the current statistic
        - windowsize_lagged_statistics (*int*): the windowsize for the feature engineering of the lagged statistics
        - test_year (*int*): the year that should be used as test set
        - datatype (*str*): if the data is in american or german type
        - date_column (*str*): the name of the column containg the date
        - group (*str*): if the data is from the old or API group
        - seasonal_periods (*int*): how many datapoints one season has
        - imputation (*bool*): whether to perfrom imputation or not
        - holiday_school_column (*str*): the column name containing the school holidays
        - holiday_public_column (*str*): the column name containing the public holidays
        - special_days (*list<str>*): the special days in your data
        - resample_weekly (*bool*): whether to resample weekly or not

    :param data_dir: data directory where the data is stored
    :param data: the dataset that you want to use
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param target_column: the target column for the prediction
    :param windowsize_current_statistics: the windowsize for the feature engineering of the current statistic
    :param windowsize_lagged_statistics: the windowsize for the feature engineering of the lagged statistics
    :param imputation_method: the imputation method to use. Options are: 'mean' , 'knn' , 'iterative'
    :param config: the information from dataset_specific_config.ini
    :param test_year: the year that should be used as test set
    """

    def __init__(self, data_dir: str, data: str, config_file: str, test_set_size_percentage: int, target_column: str,
                 windowsize_current_statistics: int, windowsize_lagged_statistics: int, imputation_method: str = 'None',
                 config: configparser.ConfigParser = None, test_year: int = None, event_lags: int = None):
        self.target_column = target_column
        self.data_dir = data_dir
        self.data = data
        self.windowsize_current_statistics = windowsize_current_statistics
        self.windowsize_lagged_statistics = windowsize_lagged_statistics
        self.test_year = test_year
        self.event_lags = event_lags

        self.values_for_counter = config[config_file]['values_for_counter'].replace(" ", "").split(',')
        if '' in self.values_for_counter:
            self.values_for_counter = None
        self.columns_for_counter = config[config_file]['columns_for_counter'].replace(" ", "").split(',')
        if '' in self.columns_for_counter:
            self.columns_for_counter = [None]
        self.columns_for_lags = config[config_file]['columns_for_lags'].replace(" ", "").split(',')
        if '' in self.columns_for_lags:
            self.columns_for_lags = [None]
        self.columns_for_rolling_mean = config[config_file]['columns_for_rolling_mean'].replace(" ", "").split(',')
        if '' in self.columns_for_rolling_mean:
            self.columns_for_rolling_mean = [None]
        self.columns_for_lags_rolling_mean = \
            config[config_file]['columns_for_lags_rolling_mean'].replace(" ", "").split(',')
        if '' in self.columns_for_lags_rolling_mean:
            self.columns_for_lags_rolling_mean = [None]
        self.string_columns = config[config_file]['string_columns'].replace(" ", "").split(',')
        if '' in self.string_columns:
            self.string_columns = [None]
        self.float_columns = config[config_file]['float_columns'].replace(" ", "").split(',')
        self.time_column = config[config_file]['time_column']
        self.seasonal_periods = config[config_file].getint('seasonal_periods')
        self.featuresets_regex = config[config_file]['featuresets_regex'].replace(" ", "").split(',')
        if '' in self.featuresets_regex:
            self.featuresets_regex = [None]
        self.imputation = config[config_file].getboolean('imputation')
        self.resample_weekly = config[config_file].getboolean('resample_weekly')
        self.time_format = config[config_file]['time_format']
        self.features = config[config_file]['features'].replace(" ", "").split(',')
        self.categorical_columns = config[config_file]['categorical_columns'].replace(" ", "").split(',')
        if '' in self.categorical_columns:
            self.categorical_columns = [None]
        self.max_seasonal_lags = config[config_file].getint('max_seasonal_lags')

        #  check if data is already preprocessed. If not, preprocess the data
        if os.path.exists(os.path.join(data_dir, data + '.h5')):
            featuresets = list()
            with pd.HDFStore(os.path.join(data_dir, data + '.h5'), 'r') as hdf:
                keys = hdf.keys()
                for key in keys:
                    featureset = pd.read_hdf(os.path.join(data_dir, data + '.h5'), key=key)
                    featureset.name = key[1:]
                    featuresets.append(featureset)
            if target_column in featuresets[0]:
                print('---Dataset is already preprocessed---')
            else:
                raise Exception('Dataset was already preprocessed, but with another target column. '
                                'Please check target column again.')

        else:
            print('---Start preprocessing data---')

            # load raw data
            dataset_raw = self.load_raw_data(data_dir=data_dir, data=data)

            # drop sales columns that are not target column and not useful columns
            dataset_raw = self.drop_non_target_useless_columns(df=dataset_raw)

            if self.imputation:
                dataset_raw = self.impute_dataset_train_test(df=dataset_raw,
                                                             test_set_size_percentage=test_set_size_percentage,
                                                             imputation_method=imputation_method)

            # set specific columns to datatype string
            self.set_dtypes(df=dataset_raw)

            # add features, resample, and preprocess
            featuresets = self.featureadding_and_resampling(df=dataset_raw)
            print('---Data preprocessed---')

        self.featuresets = featuresets

    def load_raw_data(self, data_dir: str, data: str) -> pd.DataFrame:
        """
        Load raw datasets

        :param data_dir: directory where the data is stored
        :param data: which dataset should be loaded

        :return: list of datasets to use for optimization
        """
        # load raw dataset
        dataset_raw = pd.read_csv(os.path.join(data_dir, data + '.csv'), index_col=self.time_column)
        if self.time_format == 'D':
            dataset_raw.index = pd.to_datetime(dataset_raw.index, format='%Y-%m-%d')
        if self.time_format == 'H':
            dataset_raw.index = pd.to_datetime(dataset_raw.index, format = '%Y/%m/%d %H:%M:%S')

        return dataset_raw

    def drop_non_target_useless_columns(self, df: pd.DataFrame):
        """
        Drop the possible target columns that where not chosen as target column

        :param df: DataFrame to use for dropping

        :return: DataFrame with only the target column and features left
        """
        return df[self.features]

    def set_dtypes(self, df: pd.DataFrame):
        """
        Function setting dtypes of dataset. cols_to_str are converted to string, rest except date to float.
        Needed due to structure of raw file

        :param df: DataFrame whose columns data types should be set
        """
        for col in df.columns:
            if col in self.string_columns:
                df[col] = df[col].astype(dtype='string')
            elif col != self.float_columns:
                df[col] = df[col].astype(dtype='float')

    def impute_dataset_train_test(self, df: pd.DataFrame = None, test_set_size_percentage: float = 20,
                                  imputation_method: str = None) -> pd.DataFrame:
        """
        Get imputed dataset as well as train and test set (fitted to train set)

        :param df: dataset to impute
        :param test_set_size_percentage: the size of the test set in percentage
        :param imputation_method: specify the used method if imputation is applied

        :return: imputed dataset, train and test set
        """
        cols_to_impute = df.loc[:, df.isna().any()].select_dtypes(exclude=['string', 'object']).columns.tolist()
        if len(cols_to_impute) == 0:
            return df
        cols_to_add = [col for col in df.columns.tolist() if col not in cols_to_impute]

        if test_set_size_percentage == 'yearly':
            test = df.loc[str(self.test_year) + '-01-01': str(self.test_year) + '-12-31']
            train_val = pd.concat([df, test]).drop_duplicates(keep=False)
        else:
            train_val, _ = train_test_split(df, test_size=test_set_size_percentage * 0.01, random_state=42,
                                            shuffle=False)

        if imputation_method == 'mean':
            imputer = get_simple_imputer(df=train_val.filter(cols_to_impute))
        elif imputation_method == 'knn':
            imputer = get_knn_imputer(df=train_val.filter(cols_to_impute))
        else:
            imputer = get_iter_imputer(df=train_val.filter(cols_to_impute))
        data = imputer.transform(X=df.filter(cols_to_impute))
        dataset_imp = pd.concat([pd.DataFrame(data=data,
                                              columns=cols_to_impute, index=df.index), df[cols_to_add]],
                                axis=1, sort=False)
        return dataset_imp

    def featureadding_and_resampling(self, df: pd.DataFrame) -> list:
        """
        Function preparing train and test sets for training based on raw dataset:
        - Feature Extraction
        (- Resampling if specified)
        - Deletion of non-target sales columns

        :param df: dataset with raw samples

        :return: Data with added features and resampling
        """
        # check if dataset is long enough for the given number of seasonal lags
        seasonal_lags = (len(df)//self.seasonal_periods) - 6
        if seasonal_lags < 0:
            seasonal_lags = 0
        elif seasonal_lags > self.max_seasonal_lags:
            seasonal_lags = self.max_seasonal_lags

        print('--Adding calendar dataset--')
        FeatureAdder.add_cal_features(df=df, columns_for_counter=self.columns_for_counter, event_lags=self.event_lags,
                                      resample_weekly=self.resample_weekly, values_for_counter=self.values_for_counter)
        print('--Added calendar dataset--')

        if not self.resample_weekly:
            print('-Adding statistical dataset-')
            FeatureAdder.add_statistical_features(seasonal_periods=self.seasonal_periods,
                                                  windowsize_current_statistics=self.windowsize_current_statistics,
                                                  windowsize_lagged_statistics=self.windowsize_lagged_statistics,
                                                  seasonal_lags=seasonal_lags, df=df,
                                                  resample_weekly=self.resample_weekly,
                                                  columns_for_lags=self.columns_for_lags,
                                                  columns_for_lags_rolling_mean=self.columns_for_lags_rolling_mean,
                                                  columns_for_rolling_mean=self.columns_for_rolling_mean,)

        # one hot encode the data
        print('-one-hot-encoding the data-')
        if None in self.categorical_columns:
            df = get_one_hot_encoded_df(df=df, columns_to_encode=list(df.select_dtypes(include=['string']).columns))
        else:
            df = get_one_hot_encoded_df(df=df, columns_to_encode=list(
                df.select_dtypes(include=['string']).columns) + self.categorical_columns)
        print('-one-hot-encoded the data-')

        # resample
        if self.resample_weekly:
            print('-Weekly resample data-')
            df = df.resample('W').apply(lambda x: custom_resampler(arraylike=x, target_column=self.target_column))
            if 'cal_date_weekday' in df.columns:
                drop_columns(df=df, columns=['cal_date_weekday'])
            if 'cal_date_weekday_sin' in df.columns:
                drop_columns(df=df, columns=['cal_date_weekday_sin', 'cal_date_weekday_cos'])
            df = df.loc[:str(self.test_year) + '-12-31']
            print('-Weekly resampled data-')

            # statistical feature extraction on dataset
            print('-Adding statistical dataset-')
            FeatureAdder.add_statistical_features(seasonal_periods=self.seasonal_periods,
                                                  columns_for_lags=self.columns_for_lags,
                                                  columns_for_lags_rolling_mean=self.columns_for_lags_rolling_mean,
                                                  columns_for_rolling_mean=self.columns_for_rolling_mean,
                                                  windowsize_current_statistics=self.windowsize_current_statistics,
                                                  windowsize_lagged_statistics=self.windowsize_lagged_statistics,
                                                  seasonal_lags=seasonal_lags, df=df,
                                                  resample_weekly=self.resample_weekly)

        # drop columns that stay constant
        drop_columns(df=df, columns=df.columns[df.nunique() <= 1])

        # drop missing values after adding statistical features (e.g. due to lagged features)
        df.dropna(inplace=True)

        featureset_full = df.copy()
        featureset_full.name = 'dataset_full'

        filename_h5 = os.path.join(self.data_dir, self.data + '.h5')
        featureset_full.to_hdf(filename_h5, key='dataset_full')

        featuresets = []
        featuresets.append(featureset_full)

        if not None in self.featuresets_regex:
            for featureset_regex in self.featuresets_regex:
                featureset = pd.concat([df[self.target_column], df.filter(regex=featureset_regex)], axis=1)
                featureset.name = 'featureset_' + featureset_regex
                featureset.to_hdf(filename_h5, key='featureset_' + featureset_regex)
                featuresets.append(featureset)

        return featuresets
