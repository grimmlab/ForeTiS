from . import _baseline_model

import pandas as pd
import numpy as np


class AverageSeasonal(_baseline_model.BaselineModel):
    """
    Implementation of a class for AverageSeasonal.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self):
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        self.window = self.suggest_hyperparam_to_optuna('window')
        return AverageSeasonal

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

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for the AverageSeasonal model.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        observed_period = retrain.shift(self.datasets.seasonal_periods)
        observed_period = observed_period.tail(self.window) if hasattr(self, 'window') else retrain
        self.average = observed_period[self.target_column].mean()

        if self.prediction is not None:
            if len(observed_period[self.target_column]) > len(self.prediction):
                residuals = observed_period[self.target_column][-len(self.prediction):] - self.prediction
            else:
                residuals = observed_period[self.target_column] - \
                            self.prediction[-len(observed_period[self.target_column]):]
        else:
            residuals = 0
        var = np.quantile(residuals, 0.68)
        self.var = var**2

    def update(self, update: pd.DataFrame, period: int):
        """
        Implementation of the retraining for the AverageSeasonal model.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        observed_period = update.shift(self.datasets.seasonal_periods).tail(
            self.window) if hasattr(self, 'window') else update
        self.average = observed_period[self.target_column].mean()

        residuals = observed_period[self.target_column][-len(self.prediction):] - self.prediction
        var = np.quantile(residuals, 0.68)
        self.var = var ** 2
