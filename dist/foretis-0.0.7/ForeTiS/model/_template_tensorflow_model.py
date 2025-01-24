import tensorflow as tf

from . import _tensorflow_model


class TemplateTensorflowModel(_tensorflow_model.TensorflowModel):
    """
    Template file for a prediction model based on :obj:`~ForeTiS.model._tensorflow_model.TensorflowModel`

    See :obj:`~ForeTiS.model._base_model.BaseModel` and :obj:`~ForeTiS.model._tensorflow_model.TensorflowModel`
    for more information on the attributes.

    **Steps you have to do to add your own model:**

        1. Copy this template file and rename it according to your model (will be the name to call it later on the command line)

        2. Rename the class and add it to *ForeTiS.model.__init__.py*

        3. Adjust the class attributes if necessary

        4. Define your model in *define_model()*

        5. Define the hyperparameters and ranges you want to use for optimization in *define_hyperparams_to_tune()*.

           CAUTION: Some hyperparameters are already defined in :obj:`~ForeTiS.model._tensorflow_model.TensorflowModel.common_hyperparams()`,
           which you can directly use here. Some of them are already suggested in :obj:`~ForeTiS.model.tensorflow_model.TensorflowModel`.

        6. Test your new prediction model using toy data
    """
