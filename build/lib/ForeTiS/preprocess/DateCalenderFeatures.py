import pandas as pd
import math

from ForeTiS.preprocess.raw_data_functions import drop_columns, encode_cyclical_features


def add_date_based_features(df: pd.DataFrame):
    """
    Function adding date based features to dataset

    :param df: dataset for adding features
    """
    df['cal_date_day_of_month'] = df.index.day
    df['cal_date_weekday'] = df.index.weekday
    df['cal_date_month'] = df.index.month
    encode_cyclical_features(df=df, columns=['cal_date_day_of_month', 'cal_date_weekday', 'cal_date_month'])


def add_counters(df: pd.DataFrame, columns_for_counter: list, resample_weekly: bool, event_lags: list,
                 values_for_counter: list):
    """
    Function adding counters for upcoming or past public holidays (according to event_lags)
    with own counters for those specified in special_days

    :param df: dataset for adding features
    :param resample_weekly: whether to resample weekly or not
    """
    if resample_weekly:
        if None not in columns_for_counter:
            for index, row in df.iterrows():
                for column in row.columns:
                    if column in columns_for_counter:
                        if row[column] in values_for_counter:
                            for lag in event_lags:
                                if (index + pd.Timedelta(days=lag)) in df.index:
                                    df.at[index + pd.Timedelta(days=lag), 'cal_' + column + '_Counter'] = -math.ceil(lag/7)
    else:
        if None not in columns_for_counter:
            for index, row in df.iterrows():
                for item in row.items():
                    if item[0] in columns_for_counter:
                        if item[1] in values_for_counter:
                            for lag in event_lags:
                                if (index+pd.Timedelta(days=lag)) in df.index:
                                    df.at[index+pd.Timedelta(days=lag), 'cal_' + item + '_Counter'] = -lag
    if None not in columns_for_counter:
        drop_columns(df=df, columns=columns_for_counter)
    df[[col for col in df.columns if 'Counter' in col]] = \
        df[[col for col in df.columns if 'Counter' in col]].fillna(value=99)

