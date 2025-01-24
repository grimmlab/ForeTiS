import sklearn

from . import _sklearn_model


class BayesianRidge(_sklearn_model.SklearnModel):
    """
    Implementation of a class for BayesianRidge.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self) -> sklearn.linear_model.BayesianRidge:
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined for XGBoost are suggested for optimization
        self.conf = True

        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()

        alpha_1 = self.suggest_hyperparam_to_optuna('alpha_1')
        alpha_2 = self.suggest_hyperparam_to_optuna('alpha_2')
        lambda_1 = self.suggest_hyperparam_to_optuna('lambda_1')
        lambda_2 = self.suggest_hyperparam_to_optuna('lambda_2')
        params = {}
        params.update({'fit_intercept': True})
        params.update({'n_iter': 10000})
        params.update({'tol': 1e-3})
        params.update({'copy_X': True})
        params.update({'verbose': False})
        params.update({'compute_score': False})
        params.update({'alpha_init': None})
        params.update({'lambda_init': None})
        return sklearn.linear_model.BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                                                  lambda_2=lambda_2, **params)

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
            'standardize_X': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }
