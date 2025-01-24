import torch
import sklearn
import pandas as pd
import numpy as np

from . import _torch_model
from ._model_classes import GetOutputZero, PrepareForlstm, PrepareForDropout
from blitz.modules import BayesianLSTM, BayesianLinear


class LSTMbayes(_torch_model.TorchModel):
    """
    Implementation of a class for a bayesian Long Short-Term Memory (LSTM) network.

    See :obj:`~ForeTiS.model._base_model.BaseModel` and :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on the attributes.
    """

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of a bayesian LSTM network.

        Architecture:
            - Bayesian LSTM, Dropout, Linear
            - Bayesian Linear output layer

        Number of output channels of the first layer, dropout rate, frequency of a doubling of the output channels and
        number of units in the first linear layer. may be fixed or optimized.
        """
        self.conf = True
        self.num_monte_carlo = self.suggest_hyperparam_to_optuna('num_monte_carlo')

        self.y_scaler = sklearn.preprocessing.StandardScaler()
        self.sequential = True
        self.seq_length = self.suggest_hyperparam_to_optuna('seq_length')
        model = []
        p = self.suggest_hyperparam_to_optuna('dropout')
        n_feature = self.dataset.shape[1]
        lstm_hidden_dim = self.suggest_hyperparam_to_optuna('lstm_hidden_dim')

        bias = self.suggest_hyperparam_to_optuna('bias')
        prior_sigma_1 = self.suggest_hyperparam_to_optuna('prior_sigma_1')
        prior_sigma_2 = self.suggest_hyperparam_to_optuna('prior_sigma_2')
        prior_pi = self.suggest_hyperparam_to_optuna('prior_pi')
        posterior_mu_init = self.suggest_hyperparam_to_optuna('posterior_mu_init')
        posterior_rho_init = self.suggest_hyperparam_to_optuna('posterior_rho_init')
        freeze = self.suggest_hyperparam_to_optuna('freeze')
        peephole = self.suggest_hyperparam_to_optuna('peephole')

        model.append(PrepareForlstm())
        for layer in range(self.suggest_hyperparam_to_optuna('n_lstm_layers')):
            if layer == 0:
                model.append(BayesianLSTM(in_features=n_feature, out_features=lstm_hidden_dim, bias=bias,
                                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi,
                                          posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init,
                                          freeze=freeze, peephole=peephole))
                model.append(GetOutputZero())
            else:
                model.append(BayesianLSTM(in_features=lstm_hidden_dim, out_features=lstm_hidden_dim, bias=bias,
                                          prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi,
                                          posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init,
                                          freeze=freeze, peephole=peephole))
                model.append(GetOutputZero())
        model.append(PrepareForDropout())
        model.append(torch.nn.Dropout(p))
        model.append(BayesianLinear(in_features=lstm_hidden_dim, out_features=self.n_outputs))

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
            },
            'bias': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'prior_sigma_1': {
                'datatype': 'float',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'prior_sigma_2': {
                'datatype': 'float',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'prior_pi': {
                'datatype': 'float',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'posterior_mu_init': {
                'datatype': 'float',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'posterior_rho_init': {
                'datatype': 'float',
                'lower_bound': -3.0,
                'upper_bound': 3.0
            },
            'freeze': {
                'datatype': 'categorical',
                'list_of_values': [True]
            },
            'peephole': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'num_monte_carlo': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100
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
        Implementation of a prediction based on input features for the bayes lstm model.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        self.model.eval()
        predictions = None
        conf = None
        if type(X_in) == pd.DataFrame:
            dataloader = self.get_dataloader(X=X_in.drop(labels=[self.target_column], axis=1),
                                             y=X_in[self.target_column], only_transform=True, predict=True)
            with torch.no_grad():
                for inputs in dataloader:
                    inputs = inputs.view(1, self.seq_length, -1)
                    inputs = inputs.to(device=self.device)
                    predictions_mc = []
                    for _ in range(self.num_monte_carlo):
                        # with torch.autocast(device_type=self.device.type, enabled=self.enabled):
                        output = self.model(inputs)
                        predictions_mc.append(output)
                    predictions_ = torch.stack(predictions_mc)
                    outputs = torch.mean(predictions_, dim=0)
                    confidence = torch.var(predictions_, dim=0)
                    predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
                    conf = torch.clone(confidence) if conf is None else torch.cat((conf, confidence))
        else:
            inputs = X_in.reshape(1, self.seq_length, -1)
            with torch.no_grad():
                inputs = torch.tensor(inputs.astype(np.float32))
                inputs = inputs.to(device=self.device)
                predictions_mc = []
                for _ in range(self.num_monte_carlo):
                    # with torch.autocast(device_type=self.device.type, enabled=self.enabled):
                    output = self.model(inputs)
                    predictions_mc.append(output)
                predictions_ = torch.stack(predictions_mc)
                outputs = torch.mean(predictions_, dim=0)
                confidence = torch.var(predictions_, dim=0)
                predictions = torch.clone(outputs)
                conf = torch.clone(confidence)
        self.prediction = self.y_scaler.inverse_transform(predictions.cpu().detach().numpy()).flatten()
        conf = self.y_scaler.inverse_transform(conf.cpu().detach().numpy())
        return self.prediction, self.var, conf.flatten()

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






