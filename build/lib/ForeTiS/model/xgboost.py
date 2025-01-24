import xgboost

from . import _sklearn_model


class XgBoost(_sklearn_model.SklearnModel):
    """
    Implementation of a class for XGBoost.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self) -> xgboost.XGBModel:
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined for XGBoost are suggested for optimization
        params = self.suggest_all_hyperparams_to_optuna()
        params.update({'random_state': 42})
        params.update({'verbosity': 0})
        params.update({'objective': 'reg:squarederror'})
        params.update({'tree_method': 'auto'})
        return xgboost.XGBRegressor(**params)


    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'n_estimators': {
                'datatype': 'int',
                'lower_bound': 500,
                'upper_bound': 1000,
                'step': 50
            },
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 2,
                'upper_bound': 10
            },
            'learning_rate': {
                'datatype': 'float',
                'lower_bound': 0.025,
                'upper_bound': 0.3,
                'step': 0.025
            },
            'gamma': {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 1000,
                'step': 10
            },
            'subsample': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 1.0,
                'step': 0.05
            },
            'colsample_bytree': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 1.0,
                'step': 0.05
            },
            'reg_lambda': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000,
                'step': 1
            },
            'reg_alpha': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 1000,
                'step': 1
            }
        }
