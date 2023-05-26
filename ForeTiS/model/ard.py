import sklearn

from . import _sklearn_model


class ARDRegression(_sklearn_model.SklearnModel):
    """
    Implementation of a class for ARDRegression.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self) -> sklearn.linear_model.ARDRegression:
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        self.conf = True

        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()

        alpha_1 = self.suggest_hyperparam_to_optuna('alpha_1')
        alpha_2 = self.suggest_hyperparam_to_optuna('alpha_2')
        lambda_1 = self.suggest_hyperparam_to_optuna('lambda_1')
        lambda_2 = self.suggest_hyperparam_to_optuna('lambda_2')
        threshold_lambda = self.suggest_hyperparam_to_optuna('threshold_lambda')
        params = {}
        params.update({'fit_intercept': True})
        params.update({'n_iter': 10000})
        params.update({'tol': 1e-3})
        params.update({'copy_X': True})
        params.update({'verbose': False})
        params.update({'compute_score': False})
        return sklearn.linear_model.ARDRegression(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                                                  lambda_2=lambda_2, threshold_lambda=threshold_lambda, **params)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'alpha_1': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            },
            'alpha_2': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            },
            'lambda_1': {
                'datatype': 'categorical',
                'list_of_values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            },
            'lambda_2': {
                'datatype': 'categorical',
                'list_of_values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            },
            'threshold_lambda': {
                'datatype': 'categorical',
                'list_of_values': [1e2, 1e3, 1e4, 1e5, 1e6]
            },
            'standardize_X': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }
