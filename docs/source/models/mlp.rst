Multilayer Perceptron
===============================
Subsequently, we give details on our implementation of a Multilayer Perceptron (MLP, also known as feedforward neural network).
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.
We use PyTorch for our implementation. For more information on specific PyTorch objects that we use,
e.g. layers, see the `PyTorch documentation <https://pytorch.org/docs/stable/index.html>`_.

Some of the methods and attributes relevant for the MLP are already defined in its parent class `TorchModel <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_torch_model.py>`_.
There, you can e.g. find the epoch- and batch-wise training loop. In the code block below, we show the constructor of TorchModel.

    .. code-block::

        class TorchModel(_base_model.BaseModel, abc.ABC):
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

We define attributes and suggest hyperparameters that are relevant for all neural network implementations,
e.g. the ``optimizer`` to use and the ``learning_rate`` to apply.
Some attributes are also set to fixed values, for instance the loss function (``self.loss_fn``) depending on the detected machine learning task.
Furthermore, early stopping is parametrized, which we use as a measure to prevent overfitting. With early stopping,
the validation loss is monitored and if it does not improve for a certain number of epochs (``self.early_stopping_patience``),
the training process is stopped. When working with our MLP implementation, it is important to keep in mind
that some relevant code and hyperparameters can also be found in TorchModel.

The definition of the MLP model itself as well as of some specific hyperparameters and ranges can be found in the `Mlp class <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/mlp.py>`_.
In the code block below, we show its ``define_model()`` method. Our MLP model consists of ``n_layers`` of blocks, which
include a ``Linear()``, ``BatchNorm()`` and ``Dropout`` layer. The last of these blocks is followed by a ``Linear()`` output layer.
The number of outputs in the first layers is defined by a hyperparameter (``n_initial_units_factor``),
that is multiplied with the number of inputs. Then, with each of the above-mentioned blocks, the number of outputs
decreases by a percentage parameter ``perc_decrease``.
Further, we use ``Dropout`` for regularization and define the dropout rate as the hyperparameter ``p``.
Finally, we transform the list to which we added all network layers into a ``torch.nn.Sequential()`` object.

    .. code-block::

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

``self.n_outputs`` is inherited from ``BaseModel``, where it is set to 1 for the regression task (one continuous output).

Also, we implemented the Bayesian form of the MLP model which can be found in the `Mlpbayes class <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/mlpbayes.py>`_.

**References**

1. Bishop, Christopher M. (2006). Pattern recognition and machine learning. New York, Springer.
2. Goodfellow, I., Bengio, Y.,, Courville, A. (2016). Deep Learning. MIT Press. Available at https://www.deeplearningbook.org/
3. Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424, 2015.

