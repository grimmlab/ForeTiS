SARIMAX
=============================================
Subsequently, we give details on the SARIMAX approaches that are integrated in ForeTiS.
For our implementation, we use the machine learning framework statsmodels, which also provides a `user guide for these models <https://www.statsmodels.org/stable/index.html>`_.

We implemented the ARIMA method with seasonal component, called SARIMA or SARIMAX, respectively. ARIMAX is the
abbreviation for Auto-Regressive Integrated Moving Average with eXogenous variables. These models consist of
autoregressive components (AR), moving average component (MA), and a difference order I. (S)ARIMAX takes exogenous
variables into account.

Both approaches - SARIMA and SARIMAX - are currently implemented in ForeTiS.

The following code block shows the implementation of SARIMA in `sarima.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/sarima.py>`_.

    .. code-block::

            def define_model(self) -> pmdarima.ARIMA:
                """
                Definition of the actual prediction model.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                self.conf = True

                self.use_exog = False

                P = self.suggest_hyperparam_to_optuna('P')
                D = self.suggest_hyperparam_to_optuna('D')
                Q = self.suggest_hyperparam_to_optuna('Q')
                seasonal_periods = self.suggest_hyperparam_to_optuna('seasonal_periods')
                p = self.suggest_hyperparam_to_optuna('p')
                d = self.suggest_hyperparam_to_optuna('d')
                q = self.suggest_hyperparam_to_optuna('q')

                self.trend = None

                order = [p, d, q]
                seasonal_order = [P, D, Q, seasonal_periods]
                model = pmdarima.ARIMA(order=order, seasonal_order=seasonal_order, maxiter=50, disp=1, method='lbfgs',
                                       with_intercept=True, enforce_stationarity=False, suppress_warnings=True)
                return model

            def define_hyperparams_to_tune(self) -> dict:
                """
                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'p': {
                        'datatype': 'int',
                        'lower_bound': 0,
                        'upper_bound': 3
                    },
                    'd': {
                        'datatype': 'int',
                        'lower_bound': 0,
                        'upper_bound': 1
                    },
                    'q': {
                        'datatype': 'int',
                        'lower_bound': 0,
                        'upper_bound': 3
                    },
                    'P': {
                        'datatype': 'int',
                        'lower_bound': 0,
                        'upper_bound': 3
                    },
                    'D': {
                        'datatype': 'int',
                        'lower_bound': 0,
                        'upper_bound': 1
                    },
                    'Q': {
                        'datatype': 'int',
                        'lower_bound': 0,
                        'upper_bound': 3
                    },
                    'seasonal_periods': {
                        'datatype': 'categorical',
                        'list_of_values': [52]
                    }
                }

SARIMAX is implemented in a separate files containing very similar code.
Its implementation can be found in `sarimax.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/sarimax.py>`_.