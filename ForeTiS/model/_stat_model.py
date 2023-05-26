from . import _base_model
import abc
import numpy as np
import sklearn
import pandas as pd
import optuna
import statsmodels.tsa.api
from ..preprocess import raw_data_functions
from sklearn.preprocessing import PowerTransformer


class StatModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on BaseModel for all models with a statsmodels-like API to share functionalities.
    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
    """
    def __init__(self, optuna_trial: optuna.trial.Trial, datasets: list, featureset_name: str, optimize_featureset: bool,
                 current_model_name: str = None, target_column: str = None, pca_transform: bool = None):
        self.all_hyperparams = self.common_hyperparams()
        self.current_model_name = current_model_name
        super().__init__(optuna_trial=optuna_trial, datasets=datasets, featureset_name=featureset_name,
                         target_column=target_column, pca_transform=pca_transform,
                         optimize_featureset=optimize_featureset)
        self.n_features = self.featureset.shape[1]
        self.transf = self.suggest_hyperparam_to_optuna('transf')
        self.power_transformer = sklearn.preprocessing.PowerTransformer() if self.transf == 'pw' else None
        self.contains_zeros = False

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for models with statsmodels-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        retrain = self.get_transformed_set(df=retrain, target_column=self.target_column, transf=self.transf,
                                           power_transformer=self.power_transformer, only_transform=False)
        if self.current_model_name == 'es':
            # endog for exponential smoothing must be strictly positive
            if (0 in retrain[self.target_column].values) and \
                    (self.model.trend == 'mul' or self.model.seasonal == 'mul'):
                # multiplicative trend or seasonal only working with strictly-positive data
                # only done if no transform was performed, otherwise values would need to be corrected a lot
                retrain = retrain.copy()
                retrain[self.target_column] += 0.01
            retrain.index.freq = retrain.index.inferred_freq
            model = statsmodels.tsa.api.ExponentialSmoothing(endog=retrain[self.target_column],
                                                             trend=self.model.trend,
                                                             damped_trend=self.model.damped_trend,
                                                             seasonal=self.model.seasonal,
                                                             seasonal_periods=self.model.seasonal_periods)
            self.model_results = model.fit(remove_bias=self.remove_bias, use_brute=self.use_brute)
        elif 'arima' in self.current_model_name:
            if self.use_exog:
                retrain_exog = retrain.drop(labels=[self.target_column], axis=1)
                self.exog_cols_dropped = retrain_exog.columns[retrain_exog.isna().any()].tolist()
                raw_data_functions.drop_columns(retrain_exog, self.exog_cols_dropped)
                retrain_exog = retrain_exog.to_numpy(dtype=float)
                self.model.fit(y=retrain[self.target_column], X=retrain_exog, trend=self.trend)
            else:
                self.model.fit(y=retrain[self.target_column], trend=self.trend)

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
        Update existing model due to new samples.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        update = self.get_transformed_set(df=update, target_column=self.target_column, transf=self.transf,
                                          power_transformer=self.power_transformer, only_transform=False)
        if self.current_model_name == 'es':
            if (0 in update[self.target_column].values) and (self.model.trend == 'mul' or self.model.seasonal == 'mul'):
                # multiplicative trend or seasonal only working with strictly-positive data
                # only done if no transform was performed, otherwise values would need to be corrected a lot
                update = update.copy()
                update[self.target_column] += 0.01
            model = statsmodels.tsa.api.ExponentialSmoothing(endog=update[self.target_column], trend=self.model.trend,
                                                             damped_trend=self.model.damped_trend,
                                                             seasonal=self.model.seasonal,
                                                             seasonal_periods=self.model.seasonal_periods)
            self.model_results = model.fit(remove_bias=self.remove_bias, use_brute=self.use_brute)
        elif 'arima' in self.current_model_name:
            if self.use_exog:
                exog = update.drop(labels=[self.target_column], axis=1)
                raw_data_functions.drop_columns(exog, self.exog_cols_dropped)
                exog = exog.tail(period)
                exog = exog.to_numpy(dtype=float)
                self.model.update(y=update[self.target_column].tail(period), X=exog)
            else:
                self.model.update(y=update[self.target_column].tail(period))

        y_true = update[self.target_column][-len(self.prediction):]
        y_pred = self.prediction
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for models with statsmodels-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        X_in = self.get_transformed_set(df=X_in, target_column=self.target_column, transf=self.transf,
                                        power_transformer=self.power_transformer, only_transform=True)
        if self.current_model_name == 'es':
            if len(X_in) == 1:
                self.prediction = self.model_results.forecast().values
            else:
                self.prediction = self.model_results.predict(start=X_in.index[0], end=X_in.index[-1]).values
        elif 'arima' in self.current_model_name:
            n_periods = X_in.shape[0]
            exog = None
            return_conf_int = False
            if self.use_exog:
                exog = X_in.drop(labels=[self.target_column], axis=1)
                raw_data_functions.drop_columns(exog, self.exog_cols_dropped)
                exog = exog.to_numpy(dtype=float)
            if hasattr(self, 'conf'):
                return_conf_int = True
            self.prediction, conf_int = \
                self.model.predict(n_periods=n_periods, X=exog, return_conf_int=return_conf_int, alpha=0.05)
        if isinstance(self.prediction, pd.Series):
            self.prediction = self.prediction.to_numpy()
        self.prediction = self.get_inverse_transformed_set(self.prediction, self.transf, self.power_transformer)
        if hasattr(self, 'conf'):
            conf_int = self.get_inverse_transformed_set(conf_int, self.transf, self.power_transformer, is_conf=True)
            conf = (conf_int/2)**2
            return self.prediction.flatten(), self.var.flatten(), conf
        else:
            return self.prediction.flatten(), self.var.flatten()

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Implementation of a train and validation loop for models with statsmodels-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        self.prediction = None
        self.retrain(train)

        return self.predict(X_in=val)

    def get_transformed_set(self, df: pd.DataFrame, target_column: str, transf: str,
                            power_transformer: PowerTransformer, only_transform=False) -> pd.DataFrame:
        """
        Function returning dataset with (log or power) transformed column

        :param df: dataset to transform
        :param target_column: column to transform
        :param transf: type of transformation
        :param power_transformer: if power transforming was applied, the used transformer
        :param only_transform: whether to only transform or not

        :return: dataset with transformed column
        """
        dataset_manip = df.copy()
        if transf == 'pw':
            if only_transform:
                dataset_manip[target_column] = \
                    power_transformer.transform(dataset_manip[target_column].values.reshape(-1, 1))
            else:
                dataset_manip[target_column] = \
                    power_transformer.fit_transform(dataset_manip[target_column].values.reshape(-1, 1))
        if transf == 'log':
            if any(dataset_manip[target_column] < 0):
                raise NameError('Negative values for log-transform')
            if 0 in dataset_manip[target_column].values:
                self.contains_zeros = True
                dataset_manip[target_column] = np.log(dataset_manip[target_column] + 1)
            else:
                dataset_manip[target_column] = np.log(dataset_manip[target_column])
        return dataset_manip

    def get_inverse_transformed_set(self, y: np.array, transf: str, power_transformer, is_conf: bool=False) -> np.array:
        """
        Function returning inverse (log or power) transformed column

        :param y: array to be inverse transformed
        :param power_transformer: if power transforming was applied, the used transformer
        :param transf: type of transformation

        :return: transformed column
        """
        if is_conf:
            if transf == 'pw':
                y[:, 0] = power_transformer.inverse_transform(y[:, 0].reshape(-1, 1)).flatten()
                y[:, 1] = power_transformer.inverse_transform(y[:, 1].reshape(-1, 1)).flatten()
            if transf == 'log':
                y[:, 0] = np.exp(y[:, 0])
                y[:, 1] = np.exp(y[:, 1])
                if self.contains_zeros:
                    y[:, 0] -= 1
                    y[:, 1] -= 1
        else:
            if transf == 'pw':
                y = power_transformer.inverse_transform(y.reshape(-1, 1)).flatten()
            if transf == 'log':
                y = np.exp(y)
                if self.contains_zeros:
                    y -= 1

        return y

    @staticmethod
    def common_hyperparams():
        """
        Add hyperparameters that are common for PyTorch models.
        Do not need to be included in optimization for every child model.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        return {
            'transf': {
                'datatype': 'categorical',
                'list_of_values': [False, 'log', 'pw']
            }
        }
