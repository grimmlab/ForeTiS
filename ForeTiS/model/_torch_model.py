import abc
import numpy as np
import optuna
import pandas as pd
import torch.nn
import torch.utils.data
import copy
import sklearn
from blitz.losses import kl_divergence_from_nn

from . import _base_model


class TorchModel(_base_model.BaseModel, abc.ABC):
    """
    Parent class based on :obj:`~ForeTiS.model._base_model.BaseModel` for all PyTorch models to share functionalities.
    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.

    *Attributes*

        *Inherited attributes*

        See :obj:`~ForeTiS.model._base_model.BaseModel`.

        *Additional attributes*

        - batch_size (*int*): Batch size for batch-based training
        - n_epochs (*int*): Number of epochs for optimization
        - num_monte_carlo (*int*): Number of monte carlo iteration for the bayesian neural networks
        - optimizer (*torch.optim.optimizer.Optimizer*): optimizer for model fitting
        - loss_fn: loss function for model fitting
        - early_stopping_patience (*int*): epochs without improvement before early stopping
        - early_stopping_point (*int*): epoch at which early stopping occured
        - device (*torch.device*): device to use, e.g. GPU
        - X_scaler (*sklearn.preprocessing.StandardScaler*): Standard scaler for the X data

    :param optuna_trial: Trial of optuna for optimization
    :param datasets: all datasets that are available
    :param current_model_name: name of the current model according to naming of .py file in package model
    :param batch_size: batch size for neural network models
    :param n_epochs: number of epochs for neural network models
    :param target_column: the target column for the prediction
    """
    def __init__(self, optuna_trial: optuna.trial.Trial, datasets: list, featureset_name: str, optimize_featureset: bool,
                 pca_transform: bool = None, current_model_name: str = None, batch_size: int = None,
                 n_epochs: int = None, target_column: str = None):
        self.all_hyperparams = self.common_hyperparams()
        self.current_model_name = current_model_name
        super().__init__(optuna_trial=optuna_trial, datasets=datasets, featureset_name=featureset_name,
                         target_column=target_column, pca_transform=pca_transform,
                         optimize_featureset=optimize_featureset)
        self.batch_size = \
            batch_size if batch_size is not None else 2**self.suggest_hyperparam_to_optuna('batch_size_exp')
        self.n_epochs = n_epochs if n_epochs is not None else self.suggest_hyperparam_to_optuna('n_epochs')
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.suggest_hyperparam_to_optuna('learning_rate'))
        self.loss_fn = torch.nn.MSELoss()
        # early stopping if there is no improvement on validation loss for a certain number of epochs
        self.early_stopping_patience = self.suggest_hyperparam_to_optuna('early_stopping_patience')
        self.early_stopping_point = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.X_scaler = sklearn.preprocessing.StandardScaler()
        self.enabled = True

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        """
        Implementation of a train and validation loop for  PyTorch models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        self.prediction = None
        train_loader, val_loader, val = self.train_val_loader(train=train, val=val)
        best_model = copy.deepcopy(self.model)
        self.model.to(device=self.device)
        best_loss = None
        epochs_wo_improvement = 0
        scaler = torch.cuda.amp.GradScaler(enabled=False if self.device.type == 'cpu' else True)

        y_true = np.array([0])
        y_pred = np.array([0])
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

        for epoch in range(self.n_epochs):
            self.train_one_epoch(train_loader=train_loader, scaler=scaler)
            val_loss = self.validate_one_epoch(val_loader=val_loader)
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                epochs_wo_improvement = 0
                best_model = copy.deepcopy(self.model)
            else:
                epochs_wo_improvement += 1
            if epoch >= 20 and epochs_wo_improvement >= self.early_stopping_patience:
                print("Early Stopping at " + str(epoch + 1) + ' of ' + str(self.n_epochs))
                self.early_stopping_point = epoch - self.early_stopping_patience
                self.model = best_model
                return self.predict(X_in=val)
        return self.predict(X_in=val)

    def train_val_loader(self, train: pd.DataFrame, val: pd.DataFrame):
        """
        Get the Dataloader with training and validation data

        :poram train: training data
        :param val: validation data

        :return: train_loader, val_loader, val
        """
        train_loader = self.get_dataloader(X=train.drop(labels=[self.target_column], axis=1),
                                           y=train[self.target_column], only_transform=False)
        val_loader = self.get_dataloader(X=val.drop(labels=[self.target_column], axis=1), y=val[self.target_column],
                                         only_transform=True)
        return train_loader, val_loader, val

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader, scaler):
        """
        Train one epoch

        :param train_loader: DataLoader with training data
        """
        kl = 0
        self.model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device=self.device), targets.to(device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.enabled):
                outputs = self.model(inputs)
                if 'bayes' in self.current_model_name:
                    kl = kl_divergence_from_nn(self.model)
                loss = self.get_loss(outputs=outputs, targets=targets) + kl / self.batch_size
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

    def validate_one_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate one epoch

        :param val_loader: DataLoader with validation data

        :return: loss based on loss-criterion
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device=self.device), targets.to(device=self.device)
                with torch.autocast(device_type=self.device.type, enabled=self.enabled):
                    outputs = self.model(inputs)
                    total_loss += self.get_loss(outputs=outputs, targets=targets).item()
        return total_loss / len(val_loader.dataset)

    def retrain(self, retrain: pd.DataFrame):
        """
        Implementation of the retraining for PyTorch models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        retrain_loader = self.get_dataloader(X=retrain.drop(labels=[self.target_column], axis=1),
                                             y=retrain[self.target_column], only_transform=False)
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.to(device=self.device)
        scaler = torch.cuda.amp.GradScaler(enabled=False if self.device.type == 'cpu' else True)
        for epoch in range(n_epochs_to_retrain):
            self.train_one_epoch(train_loader=retrain_loader, scaler=scaler)

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
        Implementation of the retraining for PyTorch models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        update_loader = self.get_dataloader(X=update.drop(labels=[self.target_column], axis=1),
                                            y=update[self.target_column], only_transform=False)
        n_epochs_to_retrain = self.n_epochs if self.early_stopping_point is None else self.early_stopping_point
        self.model.to(device=self.device)
        scaler = torch.cuda.amp.GradScaler(enabled=False if self.device.type == 'cpu' else True)
        self.enabled = False
        for epoch in range(n_epochs_to_retrain):
            self.train_one_epoch(update_loader, scaler=scaler)

        y_true = update[self.target_column][-len(self.prediction):]
        y_pred = self.prediction
        self.var = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        dataloader = self.get_dataloader(X=X_in.drop(labels=[self.target_column], axis=1), y=X_in[self.target_column],
                                         only_transform=True, predict=True)
        self.model.eval()
        predictions = None
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.view(1, -1)
                inputs = inputs.to(device=self.device)
                # with torch.autocast(device_type=self.device.type, enabled=self.enabled):
                outputs = self.model(inputs)
                predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
        self.prediction = predictions.cpu().detach().numpy()
        return self.prediction.flatten(), self.var.flatten()

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss based on the outputs and targets

        :param outputs: outputs of the model
        :param targets: targets of the dataset

        :return: loss
        """
        if type(self.loss_fn) in [torch.nn.CrossEntropyLoss, torch.nn.NLLLoss]:
            targets = targets.long()
        return self.loss_fn(outputs, targets)

    def get_dataloader(self, X: np.array, y: np.array = None, only_transform: bool = None, predict: bool = False,
                       shuffle: bool = False) -> torch.utils.data.DataLoader:
        """
        Get a Pytorch DataLoader using the specified data and batch size

        :param X: feature matrix to use
        :param y: optional target vector to use
        :param only_transform: whether to only transform or not
        :param predict: weather to use the data for predictions or not
        :param shuffle: shuffle parameter for DataLoader

        :return: Pytorch DataLoader
        """
        # drop last sample if last batch would only contain one sample
        if (len(X) > self.batch_size) and (len(X) % self.batch_size == 1):
            X = X[:-1]
            y = y[:-1]

        if only_transform:
            X = self.X_scaler.transform(X)
        else:
            X = self.X_scaler.fit_transform(X)

        if predict:
            X, _ = np.array(X), None
        else:
            X, y = np.array(X), np.array(y)

        X = torch.tensor(X.astype(np.float32))
        y = None if predict else torch.reshape(torch.from_numpy(y).float(), (-1, 1))
        dataset = X if predict else torch.utils.data.TensorDataset(X, y)
        if predict:
            data = dataset
        else:
            data = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
                                               worker_init_fn=np.random.seed(0))
        return data

    @staticmethod
    def common_hyperparams():
        """
        Add hyperparameters that are common for PyTorch models.
        Do not need to be included in optimization for every child model.
        Also See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        return {
            'dropout': {
                'datatype': 'float',
                'lower_bound': 0,
                'upper_bound': 0.5,
                'step': 0.1
            },
            'act_function': {
                'datatype': 'categorical',
                'list_of_values': ['relu', 'tanh'] # , None, 'leakyrelu', 'sigmoid', 'softmax'
            },
            'batch_size_exp': {
                'datatype': 'int',
                'lower_bound': 3,
                'upper_bound': 6
            },
            'n_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 5
            },
            'learning_rate': {
                'datatype': 'categorical',
                'list_of_values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            },
            'early_stopping_patience': {
               'datatype': 'int',
               'lower_bound': 0,
               'upper_bound': 1000,
            }
        }

    @staticmethod
    def get_torch_object_for_string(string_to_get: str):
        """
        Get the torch object for a specific string, e.g. when suggesting to optuna as hyperparameter

        :param string_to_get: string to retrieve the torch object

        :return: torch object
        """
        string_to_object_dict = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'sigmoid': torch.nn.Sigmoid(),
            'softmax': torch.nn.Softmax()
        }
        return string_to_object_dict[string_to_get] if string_to_get is not None else None
