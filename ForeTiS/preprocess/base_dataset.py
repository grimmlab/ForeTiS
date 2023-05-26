import pathlib

import pandas as pd
import os
import configparser
from sklearn.model_selection import train_test_split

from .raw_data_functions import custom_resampler, drop_columns, get_one_hot_encoded_df, get_iter_imputer, \
    get_simple_imputer, get_knn_imputer
from . import FeatureAdder


class Dataset:
    """
    Class containing datasets ready for optimization.

    **Attributes**

        - user_input_params (*mixed*): the arguments passed by the user or default values from run.py respectively
        - values_for_counter (*list*): the values that should trigger the counter adder
        - columns_for_counter (*list*): the columns where the counter adder should be applied
        - columns_for_lags (*list*): the columns that should be lagged by one sample
        - columns_for_rolling_mean (*list*): the columns where the rolling mean should be applied
        - columns_for_lags_rolling_mean (*list*): the columns where seasonal lagged rolling mean should be applied
        - string_columns (*list*): columns containing strings
        - float_columns (*list*): columns containing floats
        - time_column (*str*): columns containing the time information
        - seasonal_periods (*int*): how many datapoints one season has
        - featuresets_regex (*list*): regular expression with which the feature sets should be filtered
        - imputation (*bool*): whether to perfrom imputation or not
        - resample_weekly (*bool*): whether to resample weekly or not
        - time_format (*str*): the time format, either "W", "D", or "H"
        - features (*list*): the features of the dataset
        - categorical_columns (*list*): the categorical columns of the dataset
        - max_seasonal_lags (*int*): maximal number of seasonal lags to be applied
        - target_column (*str*): the target column for the prediction
        - featuresets (*list*): list containing all featuresets that get created in this class

    :param data_dir: data directory where the data is stored
    :param data: the dataset that you want to use
    :param config_file_section: the section of the config file for the used dataset
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param windowsize_current_statistics: the windowsize for the feature engineering of the current statistic
    :param windowsize_lagged_statistics: the windowsize for the feature engineering of the lagged statistics
    :param imputation_method: the imputation method to use. Options are: 'mean' , 'knn' , 'iterative'
    :param config: the information from dataset_specific_config.ini
    :param event_lags: the event lags for the counters
    :param valtest_seasons: the number of seasons to be used for validation and testing when seasonal_valtest is True
    :param: seasonal_valtest: whether validation and test sets should be a multiple of the season length or a percentage of the dataset
    """

    def __init__(self, data_dir: pathlib.Path, data: str, config_file_section: str, test_set_size_percentage: int,
                 windowsize_current_statistics: int, windowsize_lagged_statistics: int, imputation_method: str = 'None',
                 config: configparser.ConfigParser = None, event_lags: int = None, valtest_seasons: int = None,
                 seasonal_valtest: bool = None):

        self.user_input_params = locals()  # distribute all handed over params in whole class

        self.values_for_counter = config[config_file_section]['values_for_counter'].replace(" ", "").split(',')
        if '' in self.values_for_counter:
            self.values_for_counter = None
        self.columns_for_counter = config[config_file_section]['columns_for_counter'].replace(" ", "").split(',')
        if '' in self.columns_for_counter:
            self.columns_for_counter = [None]
        self.columns_for_lags = config[config_file_section]['columns_for_lags'].replace(" ", "").split(',')
        if '' in self.columns_for_lags:
            self.columns_for_lags = [None]
        self.columns_for_rolling_mean = \
            config[config_file_section]['columns_for_rolling_mean'].replace(" ", "").split(',')
        if '' in self.columns_for_rolling_mean:
            self.columns_for_rolling_mean = [None]
        self.columns_for_lags_rolling_mean = \
            config[config_file_section]['columns_for_lags_rolling_mean'].replace(" ", "").split(',')
        if '' in self.columns_for_lags_rolling_mean:
            self.columns_for_lags_rolling_mean = [None]
        self.string_columns = config[config_file_section]['string_columns'].replace(" ", "").split(',')
        if '' in self.string_columns:
            self.string_columns = [None]
        self.float_columns = config[config_file_section]['float_columns'].replace(" ", "").split(',')
        self.time_column = config[config_file_section]['time_column']
        self.seasonal_periods = config[config_file_section].getint('seasonal_periods')
        self.featuresets_regex = config[config_file_section]['featuresets_regex'].replace(" ", "").split(',')
        if '' in self.featuresets_regex:
            self.featuresets_regex = [None]
        self.imputation = config[config_file_section].getboolean('imputation')
        self.resample_weekly = config[config_file_section].getboolean('resample_weekly')
        self.time_format = config[config_file_section]['time_format']
        self.features = config[config_file_section]['features'].replace(" ", "").split(',')
        self.categorical_columns = config[config_file_section]['categorical_columns'].replace(" ", "").split(',')
        if '' in self.categorical_columns:
            self.categorical_columns = [None]
        self.max_seasonal_lags = config[config_file_section].getint('max_seasonal_lags')
        self.target_column = config[config_file_section]['target_column']

        #  check if data is already preprocessed. If not, preprocess the data
        if os.path.exists(os.path.join(data_dir, data + '.h5')):
            featuresets = list()
            with pd.HDFStore(os.path.join(data_dir, data + '.h5'), 'r') as hdf:
                keys = hdf.keys()
                for key in keys:
                    featureset = pd.read_hdf(os.path.join(data_dir, data + '.h5'), key=key)
                    featureset.name = key[1:]
                    featuresets.append(featureset)
            if self.target_column in featuresets[0]:
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
            try:
                dataset_raw.index = pd.to_datetime(dataset_raw.index, format='%Y-%m-%d')
            except:
                dataset_raw.index = pd.to_datetime(dataset_raw.index, format='%d.%m.%Y')
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

        :param df: DataFrame whose columns data types should be set
        """
        for col in df.columns:
            if col in self.string_columns:
                df[col] = df[col].astype(dtype='string')
            elif col != self.float_columns:
                df[col] = df[col].astype(dtype='float')

    def impute_dataset_train_test(self, df: pd.DataFrame = None, test_set_size_percentage: int = 20,
                                  imputation_method: str = None) -> pd.DataFrame:
        """
        Get imputed dataset as well as train and test set (fitted to train set)

        :param df: dataset to impute
        :param test_set_size_percentage: the size of the test set in percentage
        :param imputation_method: specify the used method if imputation is applied

        :return: imputed dataset, train and test set
        """
        cols_to_impute = df.loc[:, df.isna().any()]
        if len(cols_to_impute) == 0:
            return df
        cols_to_add = [col for col in df.columns.tolist() if col not in cols_to_impute]

        num_cols_to_impute = cols_to_impute.select_dtypes(exclude=['string', 'object']).columns.tolist()
        str_cols_to_impute = cols_to_impute.select_dtypes(include=['string', 'object']).columns.tolist()

        if self.user_input_params['seasonal_valtest']:
            train_val = df.iloc[:-self.user_input_params['valtest_seasons']*self.seasonal_periods]
        else:
            train_val, _ = train_test_split(df, test_size=test_set_size_percentage * 0.01, random_state=42,
                                            shuffle=False)

        if len(str_cols_to_impute) > 0:
            str_imputer = get_simple_imputer(df=train_val.filter(str_cols_to_impute), strategy='most_frequent')
            str_data = str_imputer.transform(X=df.filter(str_cols_to_impute))

        if imputation_method == 'mean':
            imputer = get_simple_imputer(df=train_val.filter(num_cols_to_impute))
        elif imputation_method == 'knn':
            imputer = get_knn_imputer(df=train_val.filter(num_cols_to_impute))
        else:
            imputer = get_iter_imputer(df=train_val.filter(num_cols_to_impute))
        num_data = imputer.transform(X=df.filter(num_cols_to_impute))

        dataset_imp = pd.concat([pd.DataFrame(data=num_data, columns=num_cols_to_impute, index=df.index),
                                 pd.DataFrame(data=str_data, columns=str_cols_to_impute, index=df.index),
                                 df[cols_to_add]],
                                axis=1, sort=False)
        return dataset_imp

    def featureadding_and_resampling(self, df: pd.DataFrame) -> list:
        """
        Function preparing train and test sets based on raw dataset.

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
        FeatureAdder.add_cal_features(df=df, columns_for_counter=self.columns_for_counter, event_lags=self.user_input_params['event_lags'],
                                      resample_weekly=self.resample_weekly, values_for_counter=self.values_for_counter)
        print('--Added calendar dataset--')

        if not self.resample_weekly:
            print('-Adding statistical dataset-')
            FeatureAdder.add_statistical_features(seasonal_periods=self.seasonal_periods,
                                                  windowsize_current_statistics=self.user_input_params['windowsize_current_statistics'],
                                                  windowsize_lagged_statistics=self.user_input_params['windowsize_lagged_statistics'],
                                                  seasonal_lags=seasonal_lags, df=df,
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
            print('-Weekly resampled data-')

            # statistical feature extraction on dataset
            print('-Adding statistical dataset-')
            FeatureAdder.add_statistical_features(seasonal_periods=self.seasonal_periods,
                                                  columns_for_lags=self.columns_for_lags,
                                                  columns_for_lags_rolling_mean=self.columns_for_lags_rolling_mean,
                                                  columns_for_rolling_mean=self.columns_for_rolling_mean,
                                                  windowsize_current_statistics=self.user_input_params['windowsize_current_statistics'],
                                                  windowsize_lagged_statistics=self.user_input_params['windowsize_lagged_statistics'],
                                                  seasonal_lags=seasonal_lags, df=df)

        # drop columns that stay constant
        drop_columns(df=df, columns=df.columns[df.nunique() <= 1])

        # drop missing values after adding statistical features (e.g. due to lagged features)
        df.dropna(inplace=True)

        featureset_full = df.copy()
        featureset_full.name = 'dataset_full'

        filename_h5 = os.path.join(self.user_input_params['data_dir'], self.user_input_params['data'] + '.h5')
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
