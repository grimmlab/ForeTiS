import torch
import numpy as np
import pandas as pd
import sklearn

from . import _torch_model
from ._model_classes import GetOutputZero, PrepareForlstm, PrepareForDropout

class LSTM(_torch_model.TorchModel):
    """
    Implementation of a class for a Long Short-Term Memory (LSTM) network.

    See :obj:`~ForeTiS.model._base_model.BaseModel` and :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on the attributes.
    """

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of a LSTM network.

        Architecture:
            - LSTM, Dropout, Linear
            - Linear output layer

        Number of output channels of the first layer, dropout rate, frequency of a doubling of the output channels and
        number of units in the first linear layer. may be fixed or optimized.
        """
        self.y_scaler = sklearn.preprocessing.StandardScaler()
        self.sequential = True
        self.seq_length = self.suggest_hyperparam_to_optuna('seq_length')
        model = []
        p = self.suggest_hyperparam_to_optuna('dropout')
        n_feature = self.dataset.shape[1]
        lstm_hidden_dim = self.suggest_hyperparam_to_optuna('lstm_hidden_dim')

        model.append(PrepareForlstm())
        model.append(torch.nn.LSTM(input_size=n_feature, hidden_size=lstm_hidden_dim,
                                   num_layers=self.suggest_hyperparam_to_optuna('n_lstm_layers'), dropout=p))
        model.append(GetOutputZero())
        model.append(PrepareForDropout())
        model.append(torch.nn.Dropout(p))
        model.append(torch.nn.Linear(in_features=lstm_hidden_dim, out_features=self.n_outputs))
        return torch.nn.Sequential(*model)


    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.

        See :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """
        return {
            'lstm_hidden_dim': {
                'datatype': 'int',
                'lower_bound': 5,
                'upper_bound': 100
            },
            'seq_length': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': self.datasets.seasonal_periods
            },
            'n_lstm_layers': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 3
            }
        }

    def train_val_loader(self, train: pd.DataFrame, val: pd.DataFrame):
        train_loader = self.get_dataloader(X=train.drop(labels=[self.target_column], axis=1),
                                           y=train[self.target_column], only_transform=False)
        val_loader = self.get_dataloader(X=pd.concat([train.tail(self.seq_length), val]).
                                         drop(labels=[self.target_column], axis=1),
                                         y=pd.concat([train.tail(self.seq_length), val])[self.target_column],
                                         only_transform=True)
        val = pd.concat([train.tail(self.seq_length), val])
        return train_loader, val_loader, val

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        self.model.eval()
        predictions = None
        if isinstance(X_in, pd.DataFrame):
            dataloader = self.get_dataloader(X=X_in.drop(labels=[self.target_column], axis=1),
                                             y=X_in[self.target_column], only_transform=True, predict=True)
            with torch.no_grad():
                for inputs in dataloader:
                    inputs = inputs.view(1, self.seq_length, -1)
                    inputs = inputs.to(device=self.device)
                    # with torch.autocast(device_type=self.device.type, enabled=self.enabled):
                    outputs = self.model(inputs)
                    predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
        else:
            inputs = X_in.reshape(1, self.seq_length, -1)
            with torch.no_grad():
                inputs = torch.tensor(inputs.astype(np.float32))
                inputs = inputs.to(device=self.device)
                # with torch.autocast(device_type=self.device.type, enabled=self.enabled):
                outputs = self.model(inputs)
                predictions = torch.clone(outputs)
        self.prediction = self.y_scaler.inverse_transform(predictions.cpu().detach().numpy()).flatten()
        return self.prediction, self.var

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

        if only_transform:
            y = self.y_scaler.transform(y.values.reshape(-1, 1))
        else:
            y = self.y_scaler.fit_transform(y.values.reshape(-1, 1))
        if predict:
            X, _ = self.create_sequences(X, y)
        else:
            X, y = self.create_sequences(X, y)

        X = torch.tensor(X.astype(np.float32))
        y = None if predict else torch.reshape(torch.from_numpy(y).float(), (-1, 1))
        dataset = X if predict else torch.utils.data.TensorDataset(X, y)
        if predict:
            data = dataset
        else:
            data = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
                                               worker_init_fn=np.random.seed(0))
        return data

    def create_sequences(self, X: np.array, y: np.array) -> tuple:
        """
        Create sequenced data according to self.seq_length

        :return: sequenced data and labels
        """
        if y is not None:
            data = np.hstack((X, y))
        else:
            data = np.array(X)
        xs = []
        ys = [] if y is not None else None
        if data.shape[0] < self.seq_length:
            raise ValueError('data shorter that sequence length!')
        for i in range(data.shape[0] - self.seq_length):
            if y is not None:
                xs.append(data[i:(i + self.seq_length), :])
                ys.append(data[i + self.seq_length, -1])
            else:
                xs.append(data[i:(i + self.seq_length), :])
        return np.array(xs), np.array(ys)
