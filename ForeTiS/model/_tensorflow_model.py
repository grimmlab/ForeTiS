from . import _base_model
import abc
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import itertools
import optuna
import gpflow

from gpflow.kernels import Matern52, White, RationalQuadratic, Periodic, SquaredExponential, Polynomial


class TensorflowModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~ForeTiS.model._base_model.BaseModel` for all TensorFlow models to share functionalities.
    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.

    **Attributes**

        *Inherited attributes*

        See :obj:`~ForeTiS.model._base_model.BaseModel`.

        *Additional attributes*

        - x_scaler (*sklearn.preprocessing.StandardScaler*): Standard scaler for the x data
        - y_scaler (*sklearn.preprocessing.StandardScaler*): Standard scaler for the y data

    """

    def __init__(self, optuna_trial: optuna.trial.Trial, datasets: list, featureset_name: str, optimize_featureset: bool,
                 pca_transform: bool = None, target_column: str = None):
        self.all_hyperparams = self.common_hyperparams()
        self.conf = True
        super().__init__(optuna_trial=optuna_trial, datasets=datasets, featureset_name=featureset_name,
                         target_column=target_column, pca_transform=pca_transform,
                         optimize_featureset=optimize_featureset)


    def define_model(self) -> gpflow.models.GPR:
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        self.standardize_y = self.suggest_hyperparam_to_optuna('standardize_y')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()
        if self.standardize_y:
            self.y_scaler = sklearn.preprocessing.StandardScaler()

        optimizer_dict = {'Scipy': gpflow.optimizers.Scipy()}
        optimizer_key = self.suggest_hyperparam_to_optuna('optimizer')
        self.optimizer = optimizer_dict[optimizer_key]

        mean_function_dict = {'Constant': gpflow.mean_functions.Constant(),
                              None: None}
        mean_function_key = self.suggest_hyperparam_to_optuna('mean_function')
        self.mean_function = mean_function_dict[mean_function_key]
        kernel_key = self.suggest_hyperparam_to_optuna('kernel')
        self.kernel = self.kernel_dict[kernel_key]
        self.noise_variance = self.suggest_hyperparam_to_optuna('noise_variance')

        return gpflow.models.GPR(data=(np.zeros((5, 1)), np.zeros((5, 1))), kernel=self.kernel,
                                 mean_function=self.mean_function, noise_variance=self.noise_variance)

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        x_train = retrain.drop(self.target_column, axis=1).values.reshape(-1, retrain.shape[1] - 1)
        y_train = retrain[self.target_column].values.reshape(-1, 1)
        if hasattr(self, 'standardize_X') and self.standardize_X:
            x_train = self.x_scaler.fit_transform(x_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.fit_transform(y_train)

        self.model.data = (tf.convert_to_tensor(value=x_train.astype(float), dtype=tf.float64),
                           tf.convert_to_tensor(value=y_train.astype(float), dtype=tf.float64))
        self.optimizer.minimize(self.model.training_loss, self.model.trainable_variables)

        if self.prediction is not None:
            if len(retrain[self.target_column]) > len(self.prediction):
                y_true = retrain[self.target_column][-len(self.prediction):]
                y_pred = self.prediction
            else:
                y_true = retrain[self.target_column]
                y_pred = self.prediction[-len(retrain[self.target_column]):]
        else:
            y_true = np.array([0])
            y_pred = np.array([0])
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def update(self, update: pd.DataFrame, period: int):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        x_train = update.drop(self.target_column, axis=1).values.reshape(-1, update.shape[1] - 1)
        y_train = update[self.target_column].values.reshape(-1, 1)
        if hasattr(self, 'standardize_X') and self.standardize_X:
            x_train = self.x_scaler.fit_transform(x_train)
        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.fit_transform(y_train)

        self.model.data = (tf.convert_to_tensor(value=x_train.astype(float), dtype=tf.float64),
                           tf.convert_to_tensor(value=y_train.astype(float), dtype=tf.float64))
        self.optimizer.minimize(self.model.training_loss, self.model.trainable_variables)

        if hasattr(self, 'standardize_y') and self.standardize_y:
            y_train = self.y_scaler.inverse_transform(y_train)

        y_true = y_train[-len(self.prediction):]
        y_pred = self.prediction
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        X_in = X_in.drop(self.target_column, axis=1).values.reshape(-1, X_in.shape[1] - 1)
        if hasattr(self, 'standardize_X') and self.standardize_X:
            X_in = self.x_scaler.transform(X_in)
        predict, conf = self.model.predict_y(Xnew=tf.convert_to_tensor(value=X_in.astype(float), dtype=tf.float64))
        conf = conf.numpy()
        self.prediction = predict.numpy()
        if hasattr(self, 'standardize_y') and self.standardize_y:
            self.prediction = self.y_scaler.inverse_transform(predict)
            conf = self.y_scaler.inverse_transform(conf)
        return self.prediction.flatten(), self.var.flatten(), conf[:, 0]

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Implementation of a train and validation loop for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        # train model
        self.prediction = None
        self.retrain(retrain=train)
        # validate model
        return self.predict(X_in=val)

    def extend_kernel_combinations(self):
        """
        Function extending kernels list with combinations based on base_kernels
        """
        kernels = []
        base_kernels = ['SquaredExponential', 'Matern52', 'WhiteKernel', 'RationalQuadratic', 'Polynomial',
                        'PeriodicSquaredExponential', 'PeriodicMatern52', 'PeriodicRationalQuadratic']
        kernel_dict = {
            'SquaredExponential': SquaredExponential(),
            'WhiteKernel': White(),
            'Matern52': Matern52(),
            'RationalQuadratic': RationalQuadratic(),
            'Polynomial': Polynomial(),
            'PeriodicSquaredExponential': Periodic(SquaredExponential(), period=52),
            'PeriodicMatern52': Periodic(Matern52(), period=52),
            'PeriodicRationalQuadratic': Periodic(RationalQuadratic(), period=52)
        }
        kernels.extend(base_kernels)
        for el in list(itertools.combinations(*[base_kernels], r=2)):
            kernels.append(el[0] + '+' + el[1])
            kernel_dict[el[0] + '+' + el[1]] = kernel_dict[el[0]] + kernel_dict[el[1]]
            kernels.append(el[0] + '*' + el[1])
            kernel_dict[el[0] + '*' + el[1]] = kernel_dict[el[0]] * kernel_dict[el[1]]
        for el in list(itertools.combinations(*[base_kernels], r=3)):
            kernels.append(el[0] + '+' + el[1] + '+' + el[2])
            kernel_dict[el[0] + '+' + el[1] + '+' + el[2]] = kernel_dict[el[0]] + kernel_dict[el[1]] + kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[1] + '*' + el[2])
            kernel_dict[el[0] + '*' + el[1] + '*' + el[2]] = kernel_dict[el[0]] * kernel_dict[el[1]] * kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[1] + '+' + el[2])
            kernel_dict[el[0] + '*' + el[1] + '+' + el[2]] = kernel_dict[el[0]] * kernel_dict[el[1]] + kernel_dict[
                el[2]]
            kernels.append(el[0] + '+' + el[1] + '*' + el[2])
            kernel_dict[el[0] + '+' + el[1] + '*' + el[2]] = kernel_dict[el[0]] + kernel_dict[el[1]] * kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[2] + '+' + el[1])
            kernel_dict[el[0] + '*' + el[2] + '+' + el[1]] = kernel_dict[el[0]] * kernel_dict[el[2]] + kernel_dict[
                el[1]]
        return kernels, kernel_dict

    @staticmethod
    def common_hyperparams() -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'noise_variance': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 100,
                'log': True
            },
            'optimizer': {
                'datatype': 'categorical',
                'list_of_values': ['Scipy']
            },
            'mean_function': {
                'datatype': 'categorical',
                'list_of_values': [None, 'Constant']
            },
            'standardize_X': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'standardize_y': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            }
        }

