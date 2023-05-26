from . import _baseline_model


class AverageMoving(_baseline_model.BaselineModel):
    """
    Implementation of a class for AverageMoving.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self):
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        self.window = self.suggest_hyperparam_to_optuna('window')
        return AverageMoving

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'window': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 20
            }
        }
