import pandas as pd


def add_lagged_statistics(seasonal_periods: int, windowsize_lagged_statistics: int, seasonal_lags: int,
                          df: pd.DataFrame, columns_for_lags_rolling_mean: list):
    """
    Function adding lagged and seasonal-lagged features to dataset

    :param seasonal_periods: seasonal_period used for seasonal-lagged features
    :param windowsize_lagged_statistics: size of window used for sales statistics
    :param seasonal_lags: seasonal lags to add of the features specified
    :param df: dataset for adding features
    :param columns_for_lags_rolling_mean: the columns where seasonal lagged rolling mean should be applied
    """
    if seasonal_lags == 0:
        print('No seasonal lags defined!')
    else:
        for seasonal_lag in range(seasonal_lags):
            seasonal_lag += 1
            for feature in columns_for_lags_rolling_mean:
                # separate function as different window sizes might be interesting compared to non-seasonal
                # statistics shift by -1 so rolling stats value is calculated as the mean of the time period
                df[feature + '_seaslag' + str(seasonal_lag)] = df[feature].shift(seasonal_lag * seasonal_periods)
                df[feature + '_seaslag' + str(seasonal_lag) + '_rolling_mean' + str(windowsize_lagged_statistics)] = \
                    df[feature].shift(seasonal_lag * seasonal_periods - 1).\
                        rolling(windowsize_lagged_statistics).mean().round(13)


def add_current_statistics(seasonal_periods: int, windowsize_current_statistics: int, df: pd.DataFrame,
                           columns_for_rolling_mean: list, columns_for_lags: list):
    """
    Function adding rolling seasonal statistics

    :param seasonal_periods: seasonal_period used for seasonal rolling statistics
    :param windowsize_current_statistics: size of window used for feature statistics
    :param df: dataset for adding features
    :param columns_for_rolling_mean: the columns where the rolling mean should be applied
    :param columns_for_lags: the columns that should be lagged by one sample
    """
    if seasonal_periods <= windowsize_current_statistics:
        return
    # separate function as different window sizes might be interesting compared to non-seasonal statistics
    for feature in columns_for_lags:
        df[feature + '_lag' + str(1)] = df[feature].shift(1)
    for feature in columns_for_rolling_mean:
        df[feature + '_rolling_mean' + str(windowsize_current_statistics)] = \
            df[feature].shift(1).rolling(windowsize_current_statistics).mean()
