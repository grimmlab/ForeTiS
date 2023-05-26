import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
import gpflow
import changefinder
import optuna

from . import _tensorflow_model

class Evars_gpr(_tensorflow_model.TensorflowModel):
    """
    Implementation of a class for Gpr.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def __init__(self, optuna_trial: optuna.trial.Trial, datasets: list, featureset_name: str,
                 pca_transform: bool = None, target_column: str = None, optimize_featureset: bool = None,
                 scale_thr: float = None, scale_seasons: int = None, scale_window_factor: float = None,
                 cf_r: float = None, cf_order: int = None, cf_smooth: int = None, cf_thr_perc: int = None,
                 scale_window_minimum: int = None, max_samples_factor: int = None):
        self.__scale_seasons = scale_seasons
        self.__cf_thr_perc = cf_thr_perc
        super().__init__(optuna_trial=optuna_trial, datasets=datasets, featureset_name=featureset_name,
                         target_column=target_column, pca_transform=pca_transform,
                         optimize_featureset=optimize_featureset)
        self.__scale_thr = scale_thr if scale_thr is not None else self.suggest_hyperparam_to_optuna('scale_thr')
        self.seasonal_periods = self.datasets.seasonal_periods
        self.time_format = self.datasets.time_format
        self.scale_window = max(scale_window_minimum, int(scale_window_factor * self.seasonal_periods))
        self.__max_samples = max_samples_factor * self.seasonal_periods
        self.__cf = changefinder.ChangeFinder(r=cf_r, order=cf_order, smooth=cf_smooth)

    def get_augmented_data(self):
        """
        get augmented data

        :return: augmented dataset
        """
        samples = self.featureset.copy()[:self.change_point_index]
        samples = samples.iloc[-self.__max_samples:] if (
                    self.__max_samples is not None and samples.shape[0] > self.__max_samples) else samples
        samples_scaled = samples.copy()
        samples_scaled[self.target_column] *= self.output_scale
        return samples_scaled

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for models with sklearn-like API.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        if not hasattr(self, 'train_ind'):
            y_deseas = self.featureset[self.target_column].diff(self.seasonal_periods).dropna().values
            self.train_ind = retrain.shape[0]
            y_train_deseas = y_deseas[:self.train_ind - self.seasonal_periods]
            self.scores = []
            for i in y_train_deseas:
                self.scores.append(self.__cf.update(i))
            self.cf_threshold = np.percentile(self.scores, self.__cf_thr_perc)

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
        self.train_ind = update.shape[0]
        self.cf_threshold = np.percentile(self.scores, self.__cf_thr_perc)
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
        target_column = X_in[self.target_column]
        X_in = X_in.drop(self.target_column, axis=1)
        predictions = None
        confs = None
        cp_detected = []
        output_scale_old = 1
        self.output_scale = 1
        n_refits = 0
        for index in X_in.index:
            target = target_column.loc[index]
            sample = X_in.loc[index]
            if hasattr(self, 'standardize_X') and self.standardize_X:
                sample = self.x_scaler.transform(sample.values.reshape(1,-1))
            else:
                sample = sample.values.reshape(1,-1)
            predict, conf = self.model.predict_y(
                Xnew=tf.convert_to_tensor(value=sample.astype(float), dtype=tf.float64))
            if predictions is None:
                predictions = predict.numpy().copy()
            else:
                predictions = np.concatenate((predictions, predict.numpy()))
            if confs is None:
                confs = conf.numpy().copy()
            else:
                confs = np.concatenate((confs, conf.numpy()))
            change_point_detected = False
            try:
                y_deseas = target - \
                           self.featureset.loc[index - pd.Timedelta(self.seasonal_periods, unit=self.time_format)][
                               self.target_column]
            except (KeyError):
                y_deseas = 0
            score = self.__cf.update(y_deseas)
            self.scores.append(score)
            if score >= self.cf_threshold:
                change_point_detected = True
                curr_ind = index - pd.Timedelta(self.train_ind, unit=self.time_format)
            # Trigger remaining EVARS-GPR procedures if a change point is detected
            if change_point_detected:
                cp_detected.append(curr_ind)
                try:
                    self.change_point_index = curr_ind + pd.Timedelta(self.train_ind, unit=self.time_format)
                    mean_now = \
                        np.mean(
                            self.dataset[
                            self.change_point_index - pd.Timedelta(self.scale_window - 1, unit=self.time_format):
                            self.change_point_index][self.target_column])
                    mean_prev_seas_1 = \
                        np.mean(
                            self.dataset[
                            self.change_point_index -
                            pd.Timedelta(self.seasonal_periods + self.scale_window - 1, unit=self.time_format):
                            self.change_point_index -
                            pd.Timedelta(self.seasonal_periods, unit=self.time_format)][self.target_column])
                    mean_prev_seas_2 = \
                        np.mean(
                            self.dataset[
                            self.change_point_index -
                            pd.Timedelta(2 * self.seasonal_periods + self.scale_window - 1, unit=self.time_format):
                            self.change_point_index -
                            pd.Timedelta(2 * self.seasonal_periods + 1, unit=self.time_format)][self.target_column])
                    if self.__scale_seasons == 1 and mean_prev_seas_1 != 0:
                        self.output_scale = mean_now / mean_prev_seas_1
                    elif self.__scale_seasons == 2 and mean_prev_seas_1 != 0 and mean_prev_seas_2 != 0:
                        self.output_scale = np.mean([mean_now / mean_prev_seas_1, mean_now / mean_prev_seas_2])
                    if self.output_scale == 0:
                        raise Exception
                        # Check deviation to previous scale factor
                    if np.abs(self.output_scale - output_scale_old) / output_scale_old > self.__scale_thr:
                        n_refits += 1
                        # augment data
                        train_samples = self.get_augmented_data()
                        # retrain current model
                        self.model = gpflow.models.GPR(
                            data=(np.zeros((5, 1)), np.zeros((5, 1))), kernel=self.kernel,
                            mean_function=self.mean_function, noise_variance=self.noise_variance
                        )
                        if self.pca_transform:
                            train_samples= self.pca_transform_train_test(train_samples)
                        self.retrain(train_samples)
                        # in case of a successful refit change output_scale_old
                        output_scale_old = self.output_scale
                except Exception as exc:
                    print(exc)
        self.prediction = predictions
        if hasattr(self, 'standardize_y') and self.standardize_y:
            self.prediction = self.y_scaler.inverse_transform(predictions)
            confs = self.y_scaler.inverse_transform(confs)
        return self.prediction.flatten(), self.var.flatten(), confs[:, 0]

    def pca_transform_train_test(self, train: pd.DataFrame) -> tuple:
        """
        Deliver PCA transformed train and test set

        :param train: data for the training

        :return: tuple of transformed train and test dataset
        """
        scaler = sklearn.preprocessing.StandardScaler()
        train_stand = scaler.fit_transform(train.drop(self.target_column, axis=1))
        pca = sklearn.decomposition.PCA(0.95)
        train_transf = pca.fit_transform(train_stand)
        train_data = pd.DataFrame(data=train_transf,
                                  columns=['PC' + str(i) for i in range(train_transf.shape[1])],
                                  index=train.index)
        train_data[self.target_column] = train[self.target_column]
        return train_data

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiSHortiCo-Hortico.model._base_model.BaseModel` for more information on the format.
        """
        kernels, self.kernel_dict = self.extend_kernel_combinations()
        return {
            'scale_thr': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.5,
                'step': 0.01
            },
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': kernels,
            }
        }
