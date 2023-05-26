from . import _base_model
import abc
import pandas as pd
import numpy as np
import sklearn


class SklearnModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~ForeTiS.model._base_model.BaseModel` for all models with a sklearn-like API to share
    functionalities. See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.

    **Attributes**

        *Inherited attributes*

        See :obj:`~ForeTiS.model._base_model.BaseModel`
    """

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        x_train = retrain.drop(self.target_column, axis=1).to_numpy()
        y_train = retrain[self.target_column].to_numpy()
        if hasattr(self, 'standardize_X') and self.standardize_X:
            x_train = self.x_scaler.fit_transform(x_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
        self.model.fit(x_train, y_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.inverse_transform(y_train)

        if self.prediction is not None:
            if len(y_train) > len(self.prediction):
                y_true = y_train[-len(self.prediction):]
                y_pred = self.prediction
            else:
                y_true = y_train
                y_pred = self.prediction[-len(y_train):]
        else:
            y_true = np.array([0])
            y_pred = np.array([0])
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def update(self, update: pd.DataFrame, period: int):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        x_train = update.drop(self.target_column, axis=1).to_numpy()
        y_train = update[self.target_column].to_numpy()
        if hasattr(self, 'standardize_X') and self.standardize_X:
            x_train = self.x_scaler.fit_transform(x_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
        self.model.fit(x_train, y_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.inverse_transform(y_train)

        y_true = y_train[-len(self.prediction):]
        y_pred = self.prediction
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        X_in = X_in.drop(self.target_column, axis=1).to_numpy()
        if hasattr(self, 'standardize_X') and self.standardize_X:
            X_in = self.x_scaler.transform(X_in)

        if hasattr(self, 'conf'):
            self.prediction, std = self.model.predict(X_in, return_std=True)
        else:
            self.prediction = self.model.predict(X_in)

        if hasattr(self, 'standardize_y') and self.standardize_y:
            self.prediction = self.y_scaler.inverse_transform(self.prediction.reshape(-1, 1))
            std = self.y_scaler.inverse_transform(std.reshape(-1, 1))

        if hasattr(self, 'conf'):
            conf = std ** 2
            return self.prediction.flatten(), self.var.flatten(), conf.flatten()
        else:
            return self.prediction.flatten(), self.var.flatten()

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Implementation of a train and validation loop for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        # train model
        self.prediction = None
        self.retrain(train)
        # validate model
        return self.predict(X_in=val)
