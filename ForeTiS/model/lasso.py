import sklearn

from . import _sklearn_model


class Lasso(_sklearn_model.SklearnModel):
    """
    Implementation of a class for Lasso.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self) -> sklearn.linear_model.Lasso:
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined for XGBoost are suggested for optimization
        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()

        alpha = self.suggest_hyperparam_to_optuna('alpha')
        params = {}
        params.update({'random_state': 42})
        params.update({'fit_intercept': True})
        params.update({'copy_X': True})
        params.update({'precompute': False})
        params.update({'max_iter': 10000})
        params.update({'tol': 1e-4})
        params.update({'warm_start': False})
        params.update({'positive': False})
        params.update({'selection': 'cyclic'})
        return sklearn.linear_model.Lasso(alpha=alpha, **params)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'alpha': {
                'datatype': 'float',
                'lower_bound': 10**-3,
                'upper_bound': 10**3,
                'log': True
            },
            'standardize_X': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }
