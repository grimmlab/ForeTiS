import torch
import pandas as pd
import numpy as np

from ForeTiS.model import _torch_model
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn


class Mlp(_torch_model.TorchModel):
    """
    Implementation of a class for a bayesian feedforward Multilayer Perceptron (MLP).

    See :obj:`~ForeTiS.model._base_model.BaseModel` and :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on the attributes.
    """

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of an MLP network.

        Architecture:

            - N_LAYERS of (bayesian Linear (+ ActivationFunction) (+ BatchNorm) + Dropout)
            - Bayesian Linear output layer
            - Dropout layer

        Number of units in the first bayesian linear layer and percentage decrease after each may be fixed or optimized.
        """
        self.conf = True
        self.num_monte_carlo = self.suggest_hyperparam_to_optuna('num_monte_carlo')

        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        self.n_features = self.dataset.shape[1] - 1
        in_features = self.n_features
        out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_initial_units_factor'))
        p = self.suggest_hyperparam_to_optuna('dropout')
        perc_decrease = self.suggest_hyperparam_to_optuna('perc_decrease_per_layer')
        batch_norm = self.suggest_hyperparam_to_optuna('batch_norm')
        for layer in range(n_layers):
            model.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            if act_function is not None:
                model.append(act_function)
            if batch_norm:
                model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p))
            in_features = out_features
            out_features = int(in_features * (1-perc_decrease))
        model.append(torch.nn.Linear(in_features=in_features, out_features=self.n_outputs))

        bnn_prior_parameters = {
            "prior_mu": self.suggest_hyperparam_to_optuna('prior_mu'),
            "prior_sigma": self.suggest_hyperparam_to_optuna('prior_sigma'),
            "posterior_mu_init": self.suggest_hyperparam_to_optuna('posterior_mu_init'),
            "posterior_rho_init": self.suggest_hyperparam_to_optuna('posterior_rho_init'),
            "type": self.suggest_hyperparam_to_optuna('type'),
            "moped_enable": self.suggest_hyperparam_to_optuna('moped_enable'),
            "moped_delta": self.suggest_hyperparam_to_optuna('moped_delta')
        }
        model = torch.nn.Sequential(*model)
        dnn_to_bnn(model, bnn_prior_parameters)
        return model

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.

        See :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on hyperparameters common for all torch models.
        """
        return {
            'n_initial_units_factor': {
                # Number of units in the first linear layer in relation to the number of inputs
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 5.0,
                'step': 0.05
            },
            'perc_decrease_per_layer': {
                # Percentage decrease of the number of units per layer
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 0.5,
                'step': 0.05
            },
            'batch_norm': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'prior_mu': {
                'datatype': 'float',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            },
            'prior_sigma': {
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
            'type': {
                'datatype': 'categorical',
                'list_of_values': ['Flipout', 'Reparameterization']
            },
            'moped_enable': {
                'datatype': 'categorical',
                'list_of_values': [False, True]
            },
            'num_monte_carlo': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100
            },
            'moped_delta': {
                'datatype': 'float',
                'lower_bound': 0.0,
                'upper_bound': 1.0
            }
        }

    def predict(self, X_in: pd.DataFrame) -> np.array:
        """
        Implementation of a prediction based on input features for PyTorch models.
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information
        """
        dataloader = self.get_dataloader(X=X_in.drop(labels=[self.target_column], axis=1), y=X_in[self.target_column],
                                         only_transform=True, predict=True)
        self.model.eval()
        predictions = None
        conf = None
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.view(1, -1)
                inputs = inputs.to(device=self.device)
                predictions_mc = []
                for _ in range(self.num_monte_carlo):
                    output = self.model(inputs)
                    predictions_mc.append(output)
                predictions_ = torch.stack(predictions_mc)
                outputs = torch.mean(predictions_, dim=0)
                confidence = torch.var(predictions_, dim=0)
                predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
                conf = torch.clone(confidence) if conf is None else torch.cat((conf, confidence))
        self.prediction = predictions.cpu().detach().numpy()
        conf = conf.cpu().detach().numpy()
        return self.prediction.flatten(), self.var.flatten(), conf.flatten()
