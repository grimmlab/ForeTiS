import pmdarima

from . import _stat_model


class Arima(_stat_model.StatModel):
    """
    Implementation of a class for a (Seasonal) Autoregressive Integrated Moving Average ((S)ARIMA) model.
    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

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
                'list_of_values': [self.datasets.seasonal_periods]
            }
        }
