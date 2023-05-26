from . import _base_model
import abc
import pandas as pd
import numpy as np
import sklearn


class BaselineModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all baseline models to share functionalities
    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
    """

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for baseline models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        observed_period = retrain.tail(self.window) if hasattr(self, 'window') else retrain
        self.average = observed_period[self.target_column].mean()

        if self.prediction is not None:
            if len(observed_period[self.target_column]) > len(self.prediction):
                y_true = observed_period[self.target_column][-len(self.prediction):]
                y_pred = self.prediction
            else:
                y_true = observed_period[self.target_column]
                y_pred = self.prediction[-len(observed_period[self.target_column]):]
        else:
            y_true = np.array([0])
            y_pred = np.array([0])
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def update(self, update: pd.DataFrame, period: int):
        """
        Implementation of the retraining for baseline models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        observed_period = update.tail(self.window) if hasattr(self, 'window') else update
        self.average = observed_period[self.target_column].mean()

        y_true = observed_period[self.target_column][-len(self.prediction):]
        y_pred = self.prediction
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for baseline models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        # use average of train set for insample prediction (insample -> knowledge of whole train set)
        self.prediction = np.full((X_in.shape[0],), self.average)
        return self.prediction.flatten(), self.var.flatten()

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Implementation of a train and validation loop for baseline models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        # train model
        self.prediction = None
        self.retrain(train)
        # validate model
        return self.predict(X_in=val)