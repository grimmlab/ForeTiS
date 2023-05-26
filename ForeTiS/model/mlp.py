import torch

from . import _torch_model


class Mlp(_torch_model.TorchModel):
    """
    Implementation of a class for a feedforward Multilayer Perceptron (MLP).

    See :obj:`~ForeTiS.model._base_model.BaseModel` and :obj:`~ForeTiS.model._torch_model.TorchModel` for more information on the attributes.
    """

    def define_model(self) -> torch.nn.Sequential:
        """
        Definition of an MLP network.

        Architecture:

            - N_LAYERS of (Linear (+ ActivationFunction) (+ BatchNorm) + Dropout)
            - Linear output layer
            - Dropout layer

        Number of units in the first linear layer and percentage decrease after each may be fixed or optimized.
        """
        n_layers = self.suggest_hyperparam_to_optuna('n_layers')
        model = []
        act_function = self.get_torch_object_for_string(string_to_get=self.suggest_hyperparam_to_optuna('act_function'))
        self.n_features = self.featureset.shape[1] - 1
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
            model.append(torch.nn.Dropout(p=p))
            in_features = out_features
            out_features = int(in_features * (1-perc_decrease))
        model.append(torch.nn.Linear(in_features=in_features, out_features=self.n_outputs))

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
            }
        }
