import torch
import pandas as pd
import numpy as np

from . import _torch_model
from blitz.modules import BayesianLinear


class Mlpbayes(_torch_model.TorchModel):
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
        self.n_features = self.featureset.shape[1] - 1
        in_features = self.n_features
        out_features = int(in_features * self.suggest_hyperparam_to_optuna('n_initial_units_factor'))
        p = self.suggest_hyperparam_to_optuna('dropout')
        perc_decrease = self.suggest_hyperparam_to_optuna('perc_decrease_per_layer')
        batch_norm = self.suggest_hyperparam_to_optuna('batch_norm')

        bias = self.suggest_hyperparam_to_optuna('bias')
        prior_sigma_1 = self.suggest_hyperparam_to_optuna('prior_sigma_1')
        prior_sigma_2 = self.suggest_hyperparam_to_optuna('prior_sigma_2')
        prior_pi = self.suggest_hyperparam_to_optuna('prior_pi')
        posterior_mu_init = self.suggest_hyperparam_to_optuna('posterior_mu_init')
        posterior_rho_init = self.suggest_hyperparam_to_optuna('posterior_rho_init')
        freeze = self.suggest_hyperparam_to_optuna('freeze')

        for layer in range(n_layers):
            model.append(BayesianLinear(in_features=in_features, out_features=out_features, bias=bias,
                                        prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi,
                                        posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init,
                                        freeze=freeze))
            if act_function is not None:
                model.append(act_function)
            if batch_norm:
                model.append(torch.nn.BatchNorm1d(num_features=out_features))
            model.append(torch.nn.Dropout(p=p))
            in_features = out_features
            out_features = int(in_features * (1-perc_decrease))
        model.append(BayesianLinear(in_features=in_features, out_features=self.n_outputs, bias=bias,
                                    prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi,
                                    posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init,
                                    freeze=freeze))

        return torch.nn.Sequential(*model)

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
                'upper_bound': 0.95,
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
                'list_of_values': [True, False]
            },
            'num_monte_carlo': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100
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
                    # with torch.autocast(device_type=self.device.type):
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
