Exponential Smoothing
=============================================
Subsequently, we give details on the Exponential Smoothing approache that is integrated in ForeTiS.
For our implementation, we use the machine learning framework statsmodels, which also provides a `user guide for these models <https://www.statsmodels.org/stable/index.html>`_.

Exponential smoothing is a univariate time series forecasting method.

The following code block shows the implementation of ES in `es.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/es.py>`_.

    .. code-block::

            class Es(_stat_model.StatModel):
                """
                Implementation of a class for an Exponential Smoothing (ES) model.
                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
                """

                def define_model(self) -> statsmodels.tsa.api.ExponentialSmoothing:
                    """
                    Definition of the actual prediction model.

                    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                    """
                    self.remove_bias = self.suggest_hyperparam_to_optuna('remove_bias')
                    self.use_brute = self.suggest_hyperparam_to_optuna('use_brute')
                    endog = self.featureset[self.target_column].copy()

                    trend = self.suggest_hyperparam_to_optuna('trend')
                    damped_trend = self.suggest_hyperparam_to_optuna('damped_trend')
                    seasonal = self.suggest_hyperparam_to_optuna('seasonal')
                    seasonal_periods = self.suggest_hyperparam_to_optuna('seasonal_periods')

                    self.model_results = None

                    if endog.eq(0).any().any() and seasonal == 'mul':
                        endog += 0.01
                    endog.index.freq = endog.index.inferred_freq

                    if trend is None:
                        damped_trend = False

                    return statsmodels.tsa.api.ExponentialSmoothing(endog=endog, trend=trend, damped_trend=damped_trend,
                                                                    seasonal=seasonal, seasonal_periods=seasonal_periods)

                def define_hyperparams_to_tune(self) -> dict:
                    """
                    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                    """
                    return {
                        'trend': {
                            'datatype': 'categorical',
                            'list_of_values': ['add', 'mul', None]
                        },
                        'damped_trend': {
                            'datatype': 'categorical',
                            'list_of_values': [False, True]
                        },
                        'seasonal': {
                            'datatype': 'categorical',
                            'list_of_values': ['add', 'mul', None]
                        },
                        'seasonal_periods': {
                            'datatype': 'categorical',
                            'list_of_values': [None, 52]
                        },
                        'use_brute': {
                            'datatype': 'categorical',
                            'list_of_values': [True, False]
                        },
                        'remove_bias': {
                            'datatype': 'categorical',
                            'list_of_values': [True, False]
                        }
                    }
