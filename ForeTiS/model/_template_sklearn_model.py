import sklearn
import optuna

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
