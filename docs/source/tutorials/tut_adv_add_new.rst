HowTo: Integrate your own prediction model
==================================================
In this tutorial, we will show you how to integrate your own new prediction model into ForeTiS using an example.

Overview
""""""""""""""
The design of the model class makes ForeTiS easy extendable with new prediction models.
The subsequent figure gives an overview on its structure.

.. image:: https://raw.githubusercontent.com/grimmlab/ForeTiS/master/docs/image/classoverview.png
    :width: 600
    :alt: structure of ForeTiS.model
    :align: center

|
All prediction models based on
`BaseModel <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/_base_model.py>`_.
This base model defines some methods that are common for all prediction models as well as all methods
that each prediction model needs to implement.
ForeTiS already contains child classes of
`BaseModel <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/_base_model.py>`_
implementing some of its obligatory methods for `TensorFlow <https://www.tensorflow.org/>`_, `PyTorch <https://pytorch.org/>`_ ,
`sklearn <https://scikit-learn.org/stable/>`_ and `statsmodels <https://www.statsmodels.org/stable/>`_.
As a consequence, adding a new prediction model based on one of these four very common machine learning frameworks only
requires the definition of two attributes and implementation of two methods, which makes ForeTiS easy extendable.



An example: Integrating k-nearest-neighbors
""""""""""""""""""""""""""""""""""""""""""""""""
We provide template files for all three frameworks and focus on
`TemplateSklearnModel <https://github.com/grimmlab/ForeTiS/blob/b9b5d5e588f4201f84eca8617601081e8d034f92/ForeTiS/model/_template_sklearn_model.py>`_
for the remainder of this tutorial:

    .. code-block::

        import sklearn

        from . import _sklearn_model


        class TemplateSklearnModel(_sklearn_model.SklearnModel):
            """
            Template file for a prediction model based on :obj:`~ForeTiS.model._sklearn_model.SklearnModel`

            See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.

            **Steps you have to do to add your own model:**

                1. Copy this template file and rename it according to your model (will be the name to call it later on on the command line)

                2. Rename the class and add it to *ForeTiS.model.__init__.py*

                3. Adjust the class attributes if necessary

                4. Define your model in *define_model()*

                5. Define the hyperparameters and ranges you want to use for optimization in *define_hyperparams_to_tune()*

                6. Test your new prediction model using toy data
            """

            def define_model(self):
                """
                Definition of the actual prediction model.

                Use *param = self.suggest_hyperparam_to_optuna(PARAM_NAME_IN_DEFINE_HYPERPARAMS_TO_TUNE)* if you want to use
                the value of a hyperparameter that should be optimized.
                The function needs to return the model object.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                ...

            def define_hyperparams_to_tune(self) -> dict:
                """
                Define the hyperparameters and ranges you want to optimize.
                Caution: they will only be optimized if you add them via *self.suggest_hyperparam_to_optuna(PARAM_NAME)* in *define_model()*

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format and options.
                """
                return {
                    'example_param_1': {
                        'datatype': 'categorical',
                        'list_of_values': ['cat', 'dog', 'elephant']
                    },
                    'example_param_2': {
                        'datatype': 'float',
                        'lower_bound': 0.05,
                        'upper_bound': 0.95,
                        'step': 0.05
                    },
                    'example_param_3': {
                        'datatype': 'int',
                        'lower_bound': 1,
                        'upper_bound': 100
                    }
                }

As an example, we will integrate `k-nearest-neighbors (knn) <https://scikit-learn.org/stable/modules/neighbors.html#>`_
as a new prediction model.

First, we copy the template file into the folder containing ForeTiS's subpackage *model* and rename it to *knn.py*.
Further, we rename the class within the file to ``Knn`` and add ``"knn"`` to ``__all__`` in
`ForeTiS.model.__init__.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/__init__.py>`_.

So with updated comments (with ``:obj:`` references for linking in the auto-generated API documentation),
our file now contains the following code:

    .. code-block::

        import sklearn

        from . import _sklearn_model


        class Knn(_sklearn_model.SklearnModel):
            """
            Implementation of a class for k nearest neighbours regressor.

            def define_model(self):
                """
                Definition of the actual prediction model.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                ...

            def define_hyperparams_to_tune(self) -> dict:
                """
                Definition of hyperparameters and ranges to optimize.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                """
                ...

Now we need to define the two attributes and implement the two methods.
Further, we optimize the two hyperparameters ``n_neighbors`` and ``weights``.
These need to be suggested to optuna via ``self.suggest_hyperparam_to_optuna(PARAM_NAME`` in ``define_model()``
and defined with their ranges in ``define_hyperparams_to_tune()``
(see `here <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/_base_model.py#:~:text=def%20define_hyperparams_to_tune(self)%20%2D%3E%20dict%3A>`_
for more information regarding the format and possible options for hyperparameter definition).

    .. code-block::

        import sklearn

        from . import _sklearn_model


        class Knn(_sklearn_model.SklearnModel):
            """
            Implementation of a class for k nearest neighbours regressor.

            def define_model(self):
                """
                Definition of the actual prediction model.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                n_neighbors = self.suggest_hyperparam_to_optuna('n_neighbors')
                weights = self.suggest_hyperparam_to_optuna('weights')
                return sklearn.neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

            def define_hyperparams_to_tune(self) -> dict:
                """
                Definition of hyperparameters and ranges to optimize.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'n_neighbors': {
                        'datatype': 'int',
                        'lower_bound': 2,
                        'upper_bound': 50,
                        'step': 2
                    },
                    'weights': {
                        'datatype': 'categorical',
                        'list_of_values': ['uniform', 'distance']
                    }
                }

Now we are able to test our new prediction model with toy data by calling ``python3 -m ForeTiS.run``
with the option ``-mod knn`` (see :ref:`HowTo: Run ForeTiS using Docker`).

This example gives an overview on how to integrate your own prediction model.
Feel free to get guidance from existing prediction models as well.
We are always happy to welcome new contributors and appreciate if you help improving ForeTiS by providing your code.

