HowTo: Adjust existing prediction models and their hyperparameters
==========================================================================
Every ForeTiS prediction model based on
`BaseModel <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_base_model.py>`_
needs to implement several methods. Most of them are already implemented in
`SklearnModel <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_sklearn_model.py>`_,
`SklearnModel <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_stats_model.py>`_,
`TorchModel <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_torch_model.py>`_ and
`TensorflowModel <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_tensorflow_model.py>`_.
So if you make use of these, a prediction model only has to implement
`define_model() <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_base_model.py#L71>`_ and
`define_hyperparams_to_tune() <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_base_model.py#L88>`_.
We will therefore focus on these two methods in this tutorial.

If you want to create your own model, see :ref:`HowTo: Integrate your own prediction model`.

We already integrated several predictions models, e.g.
`RidgeRegression <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/ridge.py>`_
and `Mlp <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/mlp.py>`_, which we will use
for demonstration purposes in this HowTo.

Besides the written documentation, we recorded the tutorial video shown below with similar content.

Adjust prediction model
""""""""""""""""""""""""""
If you want to adjust the prediction model itself, you can change its definition in its implementation of ``define_model()``.
Let's discuss an example using
`RidgeRegression <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/ridge.py>`_:

    .. code-block::

        def define_model(self) -> sklearn.linear_model.Ridge:
            # Optimize if X gets standardized or not
            self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')

            # Optimize the hyperparamter alpha
            alpha = self.suggest_hyperparam_to_optuna('alpha')

            # Set some hyperparameters that should not be optimized
            params = {}
            params.update({'random_state': 42})
            params.update({'fit_intercept': True})
            params.update({'copy_X': True})
            params.update({'max_iter': None})
            params.update({'tol': 1e-3})
            params.update({'solver': 'auto'})
            return sklearn.linear_model.Ridge(alpha=alpha, **params)

You can change the ``alpha`` term that is actually used by setting the related variable to a fixed value or suggest it
as a hyperparameter for tuning (see below for information on how to add or adjust a hyperparameter or its range).
Beyond that, you could also adjust currently fixed parameters such as ``max_iter``.

Another example can be found in
`Mlp <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/mlp.py>`_:

    .. code-block::

            def define_model(self) -> torch.nn.Sequential:
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
                model.append(torch.nn.Dropout(p))
                return torch.nn.Sequential(*model)

Currently, the model consists of ``n_layers`` of a sequence of a Linear, BatchNorm and Dropout layer, finally followed by a Linear output layer.
You can easily adjust this by e.g. adding further layers or setting ``n_layers`` to a fixed value.
Furthermore, the dropout rate ``p`` is optimized during hyperparameter search and the same rate is used for each Dropout layer.
You could set this to a fixed value or suggest a different value for each Dropout layer
(e.g. by suggesting it via ``self.suggest_hyperparam_to_optuna('dropout')`` within the ``for``-loop).
Some hyperparameters are already defined in
`TorchModel.common_hyperparams() <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_torch_model.py#L196>`_
,which you can directly use here.
Furthermore, some of them are already suggested in
`TorchModel <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_torch_model.py>`_.

Beyond that, you can also change the complete architecture of the model if you prefer to do so,
maybe by copying the file and adding your changes there (see also :ref:`HowTo: Integrate your own prediction model`).

Adjust hyperparameters
"""""""""""""""""""""""""
Besides changing the model definition, you can adjust the hyperparameters that are optimized as well as their ranges.
To set a hyperparameter to a fixed value, comment its suggestion and directly set a value, as described above.
If you want to optimize a hyperparameter which is currently set to a fixed value, do it the other way round.
If the hyperparameter is not yet defined in ``define_hyperparams_to_tune()``
Under construction.
you have to add it to ``define_hyperparams_to_tune()``.

Let's have a look at an example using
`Mlp <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/mlp.py>`_:

    .. code-block::

            def define_hyperparams_to_tune(self) -> dict:
                return {
                    'n_initial_units_factor': {
                        # Number of units in the first linear layer in relation to the number of inputs
                        'datatype': 'float',
                        'lower_bound': 0.1,
                        'upper_bound': 5,
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

There are multiple options to define a hyperparameter in easyPheno, see
`define_hyperparams_to_tune() <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_base_model.py#L88>`_
for more information regarding the format.
In the example above, three parameters are optimized depending on the number of features, besides the ones which are
defined in the parent class TorchModel in
`common_hyperparams() <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_torch_model.py#L196>`_.
The method has to return a dictionary. So if you want to add a further hyperparameter, you need to add it to the dictionary
with its name as the key and a dictionary defining its characteristics such as the ``datatype`` and ``lower_bound`` in case
of a float or int as the value.
If you only want to change the range of an existing hyperparameter, you can just change the values in this method.



