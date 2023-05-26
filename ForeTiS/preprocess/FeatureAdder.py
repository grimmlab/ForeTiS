import pandas as pd

from ForeTiS.preprocess.DateCalenderFeatures import add_date_based_features, add_counters
from ForeTiS.preprocess.StatisticalFeatures import add_current_statistics, add_lagged_statistics


def add_cal_features(df: pd.DataFrame, columns_for_counter: list, resample_weekly: bool, event_lags: list,
                     values_for_counter: list):
    """
    Function adding all calendar-based features

    :param df: dataset used for adding features
    :param resample_weekly: whether to resample weekly or not
    """
    add_date_based_features(df=df)
    add_counters(df=df, columns_for_counter=columns_for_counter, resample_weekly=resample_weekly, event_lags=event_lags,
                 values_for_counter=values_for_counter)


def add_statistical_features(seasonal_periods: int, windowsize_current_statistics: int, columns_for_lags: list,
                             columns_for_lags_rolling_mean: list, columns_for_rolling_mean: list,
                             windowsize_lagged_statistics: int, seasonal_lags: int, df: pd.DataFrame):
    """Function adding all statistical features

    :param seasonal_periods: seasonality used for seasonal-based features
    :param windowsize_current_statistics: size of window used for feature statistics
    :param windowsize_lagged_statistics: size of window used for sales statistics
    :param seasonal_lags: seasonal lags to add of the features specified
    :param df: dataset used for adding features
    """
    add_lagged_statistics(seasonal_periods=seasonal_periods, windowsize_lagged_statistics=windowsize_lagged_statistics,
                          seasonal_lags=seasonal_lags, df=df,
                          columns_for_lags_rolling_mean=columns_for_lags_rolling_mean)
    add_current_statistics(seasonal_periods=seasonal_periods, columns_for_rolling_mean=columns_for_rolling_mean,
                           columns_for_lags=columns_for_lags,
                           windowsize_current_statistics=windowsize_current_statistics, df=df)
