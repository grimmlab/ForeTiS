import os

from . import helper_functions
from ..model import _torch_model, _tensorflow_model


def check_all_specified_arguments(arguments: dict):
    """
    Check all specified arguments for plausibility

    :param arguments: all arguments provided by the user
    """
    # Check existence of save_dir
    if not os.path.exists(arguments["save_dir"]):
        raise Exception("Specified save_dir " + arguments["save_dir"] + " does not exist. Please double-check.")

    # Check meaningfulness of specified values
    if not (5 <= arguments["test_set_size_percentage"] <= 30):
        raise Exception('Specified test set size in percentage ' + str(arguments["test_set_size_percentage"]) +
                        ' is invalid, has to be between 5 and 30.')
    if not (5 <= arguments["val_set_size_percentage"] <= 30):
        raise Exception('Specified validation set size in percentage '
                        + str(arguments["val_set_size_percentage"]) + ' is invalid, has to be between 5 and 30.')

    # Check spelling of datasplit and model
    if arguments["datasplit"] not in ['timeseries-cv', 'cv', 'train-val-test']:
        raise Exception('Specified datasplit ' + arguments["datasplit"] + ' is invalid, '
                        'has to be: nested-cv | cv-test | train-val-test')
    if (arguments["models"] != 'all') and \
            (any(model not in helper_functions.get_list_of_implemented_models() for model in arguments["models"])):
        raise Exception('At least one specified model in "' + str(arguments["models"]) +
                        '" not found in implemented models nor "all" specified.' +
                        ' Check spelling or if implementation exists. Implemented models: ' +
                        str(helper_functions.get_list_of_implemented_models()))

    # Only relevant for neural networks
    if any([issubclass(helper_functions.get_mapping_name_to_class()[model], _torch_model.TorchModel) or
            issubclass(helper_functions.get_mapping_name_to_class()[model],  _tensorflow_model.TensorflowModel)
            for model in arguments["models"]]):
        if arguments["batch_size"] is not None:
            if not (2**3 <= arguments["batch_size"] <= 2**8):
                raise Exception('Specified batch size ' + str(arguments["batch_size"]) +
                                ' is invalid, has to be between 8 and 256.')
        if arguments["n_epochs"] is not None:
            if not (50 <= arguments["n_epochs"] <= 1000000):
                raise Exception('Specified number of epochs ' + str(arguments["n_epochs"]) +
                                ' is invalid, has to be between 50 and 1.000.000.')